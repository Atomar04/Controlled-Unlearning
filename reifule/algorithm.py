import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from .computation_amnesiac import compute_gae_te


def atanh(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    x = torch.clamp(x, -1 + eps, 1 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mu = nn.Linear(64, act_dim)
        self.logstd = nn.Parameter(torch.zeros(act_dim))

        self.vr = nn.Linear(64, 1)
        self.vc = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor):
        feat = self.shared(obs)
        mean = self.mu(feat)

        logstd = self.logstd.expand_as(mean)
        logstd = torch.clamp(logstd, -5.0, 1.0)
        std = torch.exp(logstd)

        v_r = self.vr(feat).squeeze(-1)
        v_c = self.vc(feat).squeeze(-1)
        return mean, std, v_r, v_c


class PPOUnlearner:
    def __init__(
        self,
        env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.03,
        max_grad_norm=0.5,
        target_kl=0.02,
        ppo_epochs=6,
        batch_size=256,
        device=None,
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        obs_dim = env.single_observation_space.shape[0] if hasattr(env, "single_observation_space") else env.observation_space.shape[0]
        act_dim = env.single_action_space.shape[0] if hasattr(env, "single_action_space") else env.action_space.shape[0]

        # Force CPU by default (Tesla M10 + torch 2.x CUDA will crash)
        self.device = device or "cpu"

        self.policy = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)

        action_space = env.single_action_space if hasattr(env, "single_action_space") else env.action_space
        self.act_low = torch.tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.act_high = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)

    def _squash(self, u: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(u)  # [-1,1]
        return self.act_low + (a + 1.0) * 0.5 * (self.act_high - self.act_low)

    def _unsquash(self, a_env: torch.Tensor) -> torch.Tensor:
        a = 2.0 * (a_env - self.act_low) / (self.act_high - self.act_low) - 1.0
        return atanh(a)

    def _logp(self, mean, std, u, a_env):
        dist = Normal(mean, std)
        logp_u = dist.log_prob(u).sum(dim=-1)

        a = 2.0 * (a_env - self.act_low) / (self.act_high - self.act_low) - 1.0
        correction = torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)

        return logp_u - correction

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)

        mean, std, v_r, v_c = self.policy(obs_t)

        u = mean if deterministic else Normal(mean, std).sample()
        a_env = self._squash(u)
        logp = self._logp(mean, std, u, a_env)

        return (
            a_env.cpu().numpy(),   # [E, act]
            logp.cpu().numpy(),    # [E]
            v_r.cpu().numpy(),     # [E]
            v_c.cpu().numpy(),     # [E]
        )

    def update(self, batch, lambda_val: float):
        """
        Expects UNFLATTENED vector-env batch:
          states:        [T, E, obs_dim]
          actions:       [T, E, act_dim]
          log_probs:     [T, E]
          rewards:       [T, E]
          costs:         [T, E]
          terminated:    [T, E] bool   (for GAE)
          last_state:    [E, obs_dim]
          last_terminated:[E] bool     (for bootstrap gating)
        """
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)     # [T,E,obs]
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)  # [T,E,act]
        old_logp = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)  # [T,E]

        rewards = np.asarray(batch["rewards"], dtype=np.float32)        # [T,E]
        costs = np.asarray(batch["costs"], dtype=np.float32)            # [T,E]
        terminated = np.asarray(batch["terminated"], dtype=np.bool_)    # [T,E]

        T, E = rewards.shape

        with torch.no_grad():
            flat_states = states.reshape(T * E, -1)
            _, _, v_r_flat, v_c_flat = self.policy(flat_states)
            v_r = v_r_flat.reshape(T, E)
            v_c = v_c_flat.reshape(T, E)

            last_state = torch.as_tensor(batch["last_state"], dtype=torch.float32, device=self.device)  # [E,obs]
            _, _, last_vr, last_vc = self.policy(last_state)
            last_vr = last_vr.detach().cpu().numpy().astype(np.float32)  # [E]
            last_vc = last_vc.detach().cpu().numpy().astype(np.float32)  # [E]

            last_term = np.asarray(batch["last_terminated"], dtype=np.bool_)  # [E]
            next_vr = np.where(last_term, 0.0, last_vr)
            next_vc = np.where(last_term, 0.0, last_vc)

            adv_r = compute_gae_te(rewards, v_r, next_vr, self.gamma, self.gae_lambda, terminated).to(self.device)
            adv_c = compute_gae_te(costs,   v_c, next_vc, self.gamma, self.gae_lambda, terminated).to(self.device)

            ret_r = adv_r + v_r
            ret_c = adv_c + v_c

            # --- IMPORTANT: normalize separately, then combine (PID not washed out) ---
            adv_r_n = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
            adv_c_std = adv_c.std()
            if adv_c_std > 1e-8:
                adv_c_n = (adv_c - adv_c.mean()) / (adv_c_std + 1e-8)
            else:
                adv_c_n = adv_c - adv_c.mean()

            adv = adv_r_n - float(lambda_val) * adv_c_n

        # Flatten for minibatches
        states_f = states.reshape(T * E, -1)
        actions_f = actions.reshape(T * E, -1)
        old_logp_f = old_logp.reshape(T * E)
        adv_f = adv.reshape(T * E)
        ret_r_f = ret_r.reshape(T * E)
        ret_c_f = ret_c.reshape(T * E)

        N = states_f.shape[0]
        idx = np.arange(N)

        last_loss = 0.0
        stop_early = False

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.batch_size):
                mb = idx[start : start + self.batch_size]
                s = states_f[mb]
                a = actions_f[mb]
                oldlp = old_logp_f[mb]
                A = adv_f[mb]
                Rr = ret_r_f[mb]
                Rc = ret_c_f[mb]

                mean, std, vr_new, vc_new = self.policy(s)

                u = self._unsquash(a)
                newlp = self._logp(mean, std, u, a)

                ratio = torch.exp(newlp - oldlp)
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * A
                pi_loss = -torch.min(surr1, surr2).mean()

                vf_loss = 0.5 * ((vr_new - Rr) ** 2).mean() + 0.5 * ((vc_new - Rc) ** 2).mean()
                ent = Normal(mean, std).entropy().sum(dim=-1).mean()

                loss = pi_loss + self.vf_coef * vf_loss - self.ent_coef * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    approx_kl = (oldlp - newlp).mean().item()
                last_loss = float(loss.item())

                if abs(approx_kl) > 1.5 * self.target_kl:
                    stop_early = True
                    break

            if stop_early:
                break

        return last_loss

    def save(self, path: str):
        torch.save({"state_dict": self.policy.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["state_dict"])
