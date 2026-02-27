import numpy as np
import torch


class PIDLagrangian:
    """
    Incremental PID controller for Lagrange multiplier λ.

    λ_{k+1} = clip( λ_k + Kp*e + Ki*I + Kd*(e - e_prev), 0, λ_max )
    where e = J_C - d.
    """

    def __init__(
        self,
        goal_cost: float = 0.05,
        Kp: float = 0.1,
        Ki: float = 0.003,
        Kd: float = 0.01,
        lambda_init: float = 0.0,
        lambda_max: float = 200.0,
        integral_max: float = 5.0,
    ):
        self.d = float(goal_cost)
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)

        self.lambda_val = float(lambda_init)
        self.lambda_max = float(lambda_max)

        self.err_integral = 0.0
        self.integral_max = float(integral_max)
        self.prev_err = 0.0

    def update(self, current_cost: float) -> float:
        error = float(current_cost) - self.d
        self.err_integral = float(
            np.clip(self.err_integral + error, -self.integral_max, self.integral_max)
        )
        derivative = error - self.prev_err

        delta = (self.Kp * error) + (self.Ki * self.err_integral) + (self.Kd * derivative)
        self.lambda_val = float(np.clip(self.lambda_val + delta, 0.0, self.lambda_max))

        self.prev_err = error
        return self.lambda_val


def compute_gae_te(rewards_te, values_te, next_values_e, gamma, lam, dones_te):
    """
    Vector-env GAE over (T, E).

    rewards_te: np.ndarray [T, E]
    values_te:  torch.Tensor [T, E]
    next_values_e: np.ndarray [E] bootstrap value after last step
    dones_te:   np.ndarray [T, E] float/bool where 1.0 means TERMINAL for GAE
               (IMPORTANT: pass terminated, NOT terminated|truncated)

    Returns:
      adv_te: torch.Tensor [T, E] (cpu tensor)
    """
    rewards = np.asarray(rewards_te, dtype=np.float32)
    dones = np.asarray(dones_te, dtype=np.float32)
    next_values = np.asarray(next_values_e, dtype=np.float32)

    T, E = rewards.shape
    adv = np.zeros((T, E), dtype=np.float32)
    last_gae = np.zeros((E,), dtype=np.float32)

    v = values_te.detach().cpu().numpy().astype(np.float32)  # [T,E]

    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]                       # [E]
        v_next = next_values if (t == T - 1) else v[t + 1]  # [E]
        delta = rewards[t] + gamma * v_next * non_terminal - v[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        adv[t] = last_gae

    return torch.from_numpy(adv)
