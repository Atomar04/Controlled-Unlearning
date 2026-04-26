import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from reifule.utils import probe_artifact_path


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def export_hazard_direction(
    probe_path: str,
    out_path: str,
    include_tau: bool = False,
):
    with open(probe_path, "rb") as f:
        payload = pickle.load(f)

    if "probe" not in payload:
        raise KeyError(f"'probe' key not found in {probe_path}")

    pipe = payload["probe"]

    if not hasattr(pipe, "named_steps"):
        raise ValueError("Expected a sklearn Pipeline with named_steps.")

    if "scaler" not in pipe.named_steps or "clf" not in pipe.named_steps:
        raise ValueError("Probe pipeline must contain 'scaler' and 'clf' steps.")

    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]

    coef_std = clf.coef_[0].astype(np.float32)
    direction = coef_std / scaler.scale_.astype(np.float32)

    save_obj = {
        "direction": torch.tensor(direction, dtype=torch.float32),
    }

    if include_tau:
        intercept_raw = float(
            clf.intercept_[0] - np.sum((scaler.mean_ / scaler.scale_) * coef_std)
        )
        tau_from_probe = float(-intercept_raw)
        save_obj["tau"] = tau_from_probe

    ensure_parent(out_path)
    torch.save(save_obj, out_path)

    print(f"Saved {out_path}")
    print("direction shape:", direction.shape)
    if include_tau:
        print("tau_from_probe:", tau_from_probe)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--probe_path",
        type=str,
        default=probe_artifact_path("hazard_probe.pkl"),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=probe_artifact_path("hazard_direction.pt"),
    )
    parser.add_argument(
        "--include_tau",
        action="store_true",
        help="Also export raw-space tau reconstructed from the probe intercept.",
    )

    args = parser.parse_args()

    export_hazard_direction(
        probe_path=args.probe_path,
        out_path=args.out,
        include_tau=args.include_tau,
    )


if __name__ == "__main__":
    main()