"""Train ChuDaDi V3 with the original DMCTrainer training path.

V3 keeps the DMC Monte Carlo final-payoff trainer unchanged while using the
334-dimensional cumulative-history observation and a two-input dual-tower model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


# Running this file from the project root would otherwise import an installed
# rlcard package before the local training tree. Put ./rlcard first.
RLCARD_ROOT = Path(__file__).resolve().parents[1]
if str(RLCARD_ROOT) not in sys.path:
    sys.path.insert(0, str(RLCARD_ROOT))


DEFAULT_SAVEDIR = str(RLCARD_ROOT / "experiments" / "dmc_result")
DEFAULT_XPID = "chudadi_v3_dmc_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ChuDaDi V3 DMC using the legacy DMCTrainer path."
    )
    parser.add_argument("--rule", choices=("northern", "southern"), default="northern")
    parser.add_argument("--xpid", default=DEFAULT_XPID)
    parser.add_argument(
        "--savedir",
        default=DEFAULT_SAVEDIR,
        help="Checkpoint root. Default stays inside the rlcard folder.",
    )
    parser.add_argument("--cuda", default="0", help="CUDA_VISIBLE_DEVICES value; default uses GPU 0. Use --cuda \"\" for CPU.")
    parser.add_argument("--training-device", default="0", help="GPU index for learner, or cpu.")
    parser.add_argument("--load-model", action="store_true", help="Resume from savedir/xpid/model.tar.")
    parser.add_argument(
        "--total-frames",
        type=int,
        default=None,
        help="Absolute DMCTrainer frame target. Mutually exclusive with --phase-frames.",
    )
    parser.add_argument(
        "--phase-frames",
        type=int,
        default=None,
        help="Additional frames to train in this phase. Defaults to 1000000 when --total-frames is omitted.",
    )
    parser.add_argument("--save-interval", type=int, default=30, help="Checkpoint interval in minutes.")
    parser.add_argument("--num-actor-devices", type=int, default=1)
    parser.add_argument("--num-actors", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unroll-length", type=int, default=100)
    parser.add_argument("--num-buffers", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=40.0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.00001)
    parser.add_argument("--exp-epsilon", type=float, default=0.01)
    parser.add_argument("--log-every-learns", type=int, default=20, help="Write one CSV row per N learner updates.")
    parser.add_argument(
        "--save-seat-checkpoints",
        action="store_true",
        help="Also save per-seat .pth files at each checkpoint. model.tar is always saved.",
    )
    parser.add_argument(
        "--enable-baopei",
        action="store_true",
        help="Enable traditional Baopei scoring during training. Disabled by default for V3.",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("score", "win_loss_zero_sum", "strategic_win_zero_sum"),
        default="score",
        help="Training target. V3 defaults to raw score reward.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved config and exit without starting training.",
    )
    args = parser.parse_args()
    if args.total_frames is not None and args.phase_frames is not None:
        parser.error("Use either --total-frames or --phase-frames, not both.")
    if args.total_frames is None and args.phase_frames is None:
        args.phase_frames = 1_000_000
    if args.log_every_learns < 1:
        parser.error("--log-every-learns must be >= 1.")
    return args


def checkpoint_path(args: argparse.Namespace) -> Path:
    return Path(args.savedir).expanduser() / args.xpid / "model.tar"


def read_checkpoint_frames(path: Path, training_device: str) -> int:
    if not path.exists():
        return 0
    import torch

    map_location = f"cuda:{training_device}" if training_device != "cpu" and torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return int(checkpoint.get("frames", 0))


def resolve_total_frames(args: argparse.Namespace) -> int:
    if args.total_frames is not None:
        return args.total_frames
    current_frames = read_checkpoint_frames(checkpoint_path(args), args.training_device) if args.load_model else 0
    return current_frames + int(args.phase_frames)


def validate_device(args: argparse.Namespace) -> None:
    if args.cuda == "" or args.training_device == "cpu":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build or pass --cuda \"\" --training-device cpu."
        )


def build_env(args: argparse.Namespace):
    import rlcard

    return rlcard.make(
        "chudadi",
        config={
            "northern_rule": args.rule == "northern",
            "obs_version": "v3",
            "reward_mode": args.reward_mode,
            "enable_baopei": args.enable_baopei,
        },
    )


def config_snapshot(args: argparse.Namespace, total_frames: int, env: Any) -> dict[str, Any]:
    return {
        "rule": args.rule,
        "xpid": args.xpid,
        "savedir": str(Path(args.savedir).expanduser()),
        "checkpoint": str(checkpoint_path(args)),
        "load_model": args.load_model,
        "total_frames": total_frames,
        "phase_frames": args.phase_frames,
        "cuda": args.cuda,
        "training_device": args.training_device,
        "num_actor_devices": args.num_actor_devices,
        "num_actors": args.num_actors,
        "batch_size": args.batch_size,
        "unroll_length": args.unroll_length,
        "num_buffers": args.num_buffers,
        "num_threads": args.num_threads,
        "save_interval": args.save_interval,
        "max_grad_norm": args.max_grad_norm,
        "learning_rate": args.learning_rate,
        "alpha": args.alpha,
        "momentum": args.momentum,
        "epsilon": args.epsilon,
        "exp_epsilon": args.exp_epsilon,
        "log_every_learns": args.log_every_learns,
        "save_seat_checkpoints": args.save_seat_checkpoints,
        "reward_mode": args.reward_mode,
        "enable_baopei": args.enable_baopei,
        "model_version": "v3",
        "obs_version": "v3",
        "state_shape": env.state_shape,
        "action_shape": env.action_shape,
        "history_shape": getattr(env, "history_shape", None),
    }


def write_phase_config(args: argparse.Namespace, snapshot: dict[str, Any]) -> None:
    phase_dir = Path(args.savedir).expanduser() / args.xpid
    phase_dir.mkdir(parents=True, exist_ok=True)
    path = phase_dir / f"phase_{snapshot['rule']}_{snapshot['total_frames']}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, ensure_ascii=False)


def print_dry_run_summary(snapshot: dict[str, Any]) -> None:
    print(f"model_version={snapshot['model_version']}", flush=True)
    print(f"obs_version={snapshot['obs_version']}", flush=True)
    print(f"reward_mode={snapshot['reward_mode']}", flush=True)
    print(f"enable_baopei={str(snapshot['enable_baopei']).lower()}", flush=True)
    print(f"state_shape={snapshot['state_shape']}", flush=True)
    print(f"action_shape={snapshot['action_shape']}", flush=True)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    from rlcard.agents.dmc_agent import DMCTrainer

    validate_device(args)
    env = build_env(args)
    total_frames = resolve_total_frames(args)
    snapshot = config_snapshot(args, total_frames, env)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print_dry_run_summary(snapshot)
        return
    write_phase_config(args, snapshot)
    trainer = DMCTrainer(
        env,
        cuda=args.cuda,
        load_model=args.load_model,
        xpid=args.xpid,
        savedir=args.savedir,
        save_interval=args.save_interval,
        num_actor_devices=args.num_actor_devices,
        num_actors=args.num_actors,
        training_device=args.training_device,
        total_frames=total_frames,
        exp_epsilon=args.exp_epsilon,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        num_buffers=args.num_buffers,
        num_threads=args.num_threads,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
        momentum=args.momentum,
        epsilon=args.epsilon,
        model_version="v3",
        log_every_learns=args.log_every_learns,
        save_seat_checkpoints=args.save_seat_checkpoints,
    )
    trainer.start()


if __name__ == "__main__":
    main()
