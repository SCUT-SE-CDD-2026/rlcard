"""Fine-tune ChuDaDi V2 against a fixed V1 ONNX opponent.

This is intentionally a separate experiment script. It does not replace the
legacy DMCTrainer path. The training policy stays close to v2_gru_legacy where
possible:
- production-aligned V2 environment
- V2 GRU model/checkpoint format
- four position-specific V2 agents and optimizers loaded from model.tar
- RMSprop hyperparameters compatible with DMCTrainer
- frame-based model.tar and per-seat .pth checkpoints

V1 is used only for inference and is never updated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

RLCARD_ROOT = Path(__file__).resolve().parents[1]
if str(RLCARD_ROOT) not in sys.path:
    sys.path.insert(0, str(RLCARD_ROOT))

DEFAULT_CHECKPOINT = RLCARD_ROOT / "experiments" / "dmc_result" / "kaggle-V2" / "model.tar"
DEFAULT_V1_ONNX = RLCARD_ROOT / "experiments" / "dmc_result" / "V1-onnx" / "chudadi_v0_01.onnx"
DEFAULT_SAVEDIR = RLCARD_ROOT / "experiments" / "dmc_result"
DEFAULT_XPID = "chudadi_v2_vs_v1_onnx"
PLAYER_COUNT = 4
WIN_TARGET = 1.0
LOSE_TARGET = -1.0 / 3.0

PATTERN_GROUPS: dict[str, list[tuple[int, ...]]] = {
    "abab": [(0, 2), (1, 3)],
    "aabb": [(0, 1), (1, 2), (2, 3), (3, 0)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Fine-tune V2 GRU DMC against fixed V1 ONNX opponents.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Initial V2 model.tar checkpoint.")
    parser.add_argument("--v1-onnx", default=str(DEFAULT_V1_ONNX), help="Fixed V1 ONNX opponent path.")
    parser.add_argument("--savedir", default=str(DEFAULT_SAVEDIR), help="Output root; defaults inside rlcard.")
    parser.add_argument("--xpid", default=DEFAULT_XPID)
    parser.add_argument("--resume", action="store_true", help="Load savedir/xpid/model.tar instead of --checkpoint when present.")
    parser.add_argument("--rule", choices=("northern", "southern"), default="northern")
    parser.add_argument("--history-len", type=int, default=13)
    parser.add_argument("--cuda", default="0", help="CUDA_VISIBLE_DEVICES value. Use empty string with --training-device cpu for CPU.")
    parser.add_argument("--training-device", default="0", help="GPU index for V2 training, or cpu.")
    parser.add_argument("--total-frames", type=int, default=None, help="Absolute frame target. Mutually exclusive with --phase-frames.")
    parser.add_argument("--phase-frames", type=int, default=None, help="Additional trainable V2 transitions this phase. Defaults to 1000000.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--train-updates-per-game", type=int, default=2)
    parser.add_argument("--pattern-types", default="abab,aabb", help="Comma separated pattern groups: abab,aabb")
    parser.add_argument("--pattern-games", type=int, default=10000, help="Games per seating pattern block.")
    parser.add_argument("--save-every-frames", type=int, default=176000)
    parser.add_argument("--log-every-games", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.00001)
    parser.add_argument("--max-grad-norm", type=float, default=40.0)
    parser.add_argument("--exp-epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit without training.")
    args = parser.parse_args()
    if args.total_frames is not None and args.phase_frames is not None:
        parser.error("Use either --total-frames or --phase-frames, not both.")
    if args.total_frames is None and args.phase_frames is None:
        args.phase_frames = 1_000_000
    return args


def output_dir(args: argparse.Namespace) -> Path:
    return Path(args.savedir).expanduser() / args.xpid


def output_checkpoint(args: argparse.Namespace) -> Path:
    return output_dir(args) / "model.tar"


def selected_checkpoint(args: argparse.Namespace) -> Path:
    resume_path = output_checkpoint(args)
    if args.resume and resume_path.exists():
        return resume_path
    return Path(args.checkpoint).expanduser()


def parse_pattern_sequence(pattern_types: str) -> list[tuple[int, ...]]:
    sequence: list[tuple[int, ...]] = []
    for name in (part.strip().lower() for part in pattern_types.split(",")):
        if not name:
            continue
        if name not in PATTERN_GROUPS:
            raise ValueError(f"Unknown pattern type: {name}; expected one of {sorted(PATTERN_GROUPS)}")
        sequence.extend(PATTERN_GROUPS[name])
    if not sequence:
        raise ValueError("At least one pattern type is required")
    return sequence


def v2_seats_for_game(game_index: int, pattern_sequence: list[tuple[int, ...]], pattern_games: int) -> tuple[int, ...]:
    block = (game_index - 1) // pattern_games
    return pattern_sequence[block % len(pattern_sequence)]


class SharedSeatAgent:
    def __init__(self, agent: Any):
        self.agent = agent
        self.use_raw = getattr(agent, "use_raw", False)

    def step(self, state: dict[str, Any]) -> int:
        return self.agent.step(state)

    def eval_step(self, state: dict[str, Any]):
        return self.agent.eval_step(state)


class OnnxDmcAgent:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        from rlcard.games.chudadi.utils import card_to_id

        self.card_to_id = card_to_id
        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.obs_dim = self._dim_or_none(self.session.get_inputs()[0].shape[-1])
        self.use_raw = False
        print(f"V1 ONNX providers: {self.session.get_providers()}", flush=True)

    @staticmethod
    def _dim_or_none(value: Any) -> int | None:
        return int(value) if isinstance(value, int) else None

    def step(self, state: dict[str, Any]) -> int:
        return self.eval_step(state)[0]

    def eval_step(self, state: dict[str, Any]):
        legal_actions = state["legal_actions"]
        action_keys = list(legal_actions.keys())
        action_values = np.asarray(list(legal_actions.values()), dtype=np.float32)
        base_obs = self._legacy_obs(state) if self.obs_dim == 334 and state["obs"].shape[0] != 334 else state["obs"]
        obs = np.repeat(np.asarray(base_obs, dtype=np.float32)[None, :], len(action_keys), axis=0)
        outputs = self.session.run(None, {self.input_names[0]: obs, self.input_names[1]: action_values})
        values = np.asarray(outputs[0]).reshape(-1)
        action = action_keys[int(np.argmax(values))]
        info = {"values": {k: float(v) for k, v in zip(action_keys, values)}}
        return action, info

    def _cards_to_array(self, cards: list[Any]) -> np.ndarray:
        array = np.zeros(52, dtype=np.int8)
        for card in cards:
            array[self.card_to_id(card)] = 1
        return array

    def _legacy_obs(self, state: dict[str, Any]) -> np.ndarray:
        raw = state["raw_obs"]
        player_id = raw["current_player"]
        obs = state["obs"]
        cumulative_history = []
        for offset in (1, 2, 3):
            relative_id = (player_id + offset) % PLAYER_COUNT
            cumulative_history.append(self._cards_to_array(raw["played_cards"][relative_id]))
        return np.concatenate([obs[:172], *cumulative_history, obs[172:]])


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
            "seed": args.seed,
            "northern_rule": args.rule == "northern",
            "history_len": args.history_len,
        },
    )


def build_model(env: Any, args: argparse.Namespace):
    from rlcard.agents.dmc_agent.model import DMCModel

    device = args.training_device if args.training_device == "cpu" else str(args.training_device)
    model = DMCModel(
        env.state_shape,
        env.action_shape,
        exp_epsilon=args.exp_epsilon,
        device=device,
        model_version="v2",
        history_shape=env.history_shape,
    )
    return model


def build_optimizers(model: Any, args: argparse.Namespace):
    import torch

    return [
        torch.optim.RMSprop(
            model.parameters(position),
            lr=args.learning_rate,
            momentum=args.momentum,
            eps=args.epsilon,
            alpha=args.alpha,
        )
        for position in range(PLAYER_COUNT)
    ]


def load_training_state(model: Any, optimizers: list[Any], checkpoint: Path, training_device: str) -> tuple[int, dict[str, Any]]:
    import torch

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    map_location = f"cuda:{training_device}" if training_device != "cpu" and torch.cuda.is_available() else "cpu"
    state = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if "model_state_dict" not in state:
        raise ValueError(f"Expected DMCTrainer model.tar checkpoint with model_state_dict: {checkpoint}")
    for position in range(PLAYER_COUNT):
        model.get_agent(position).load_state_dict(state["model_state_dict"][position])
    if "optimizer_state_dict" in state:
        for position, optimizer_state in enumerate(state["optimizer_state_dict"]):
            if position < len(optimizers):
                optimizers[position].load_state_dict(optimizer_state)
    return int(state.get("frames", 0)), dict(state.get("stats", {}))


def save_training_state(
    model: Any,
    optimizers: list[Any],
    stats: dict[str, Any],
    frames: int,
    env: Any,
    args: argparse.Namespace,
) -> Path:
    import torch

    target_dir = output_dir(args)
    target_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = target_dir / "model.tar"
    torch.save(
        {
            "model_state_dict": [model.get_agent(position).state_dict() for position in range(PLAYER_COUNT)],
            "optimizer_state_dict": [optimizer.state_dict() for optimizer in optimizers],
            "stats": stats,
            "frames": frames,
            "model_version": "v2",
            "state_shape": env.state_shape,
            "action_shape": env.action_shape,
            "history_shape": env.history_shape,
        },
        checkpoint,
    )
    for position in range(PLAYER_COUNT):
        torch.save(model.get_agent(position), target_dir / f"{position}_{frames}.pth")
    return checkpoint


def transition_from_state(env: Any, state: dict[str, Any], action: int, target: float):
    import torch

    return (
        torch.from_numpy(state["obs"]).float(),
        torch.from_numpy(env.get_action_feature(action, state)).float(),
        torch.from_numpy(state["history"]).float(),
        float(target),
    )


def win_loss_targets(payoffs: list[float] | np.ndarray) -> list[float]:
    winner = int(np.argmax(payoffs))
    return [WIN_TARGET if seat == winner else LOSE_TARGET for seat in range(PLAYER_COUNT)]


def train_batch(agent: Any, optimizer: Any, replay: deque[Any], batch_size: int, device: str, max_grad_norm: float) -> float | None:
    if len(replay) < batch_size:
        return None
    import torch

    indices = torch.randint(0, len(replay), (batch_size,)).tolist()
    obs = torch.stack([replay[index][0] for index in indices]).to(device)
    actions = torch.stack([replay[index][1] for index in indices]).to(device)
    history = torch.stack([replay[index][2] for index in indices]).to(device)
    targets = torch.tensor([replay[index][3] for index in indices], dtype=torch.float32, device=device)
    values = agent.forward(obs, actions, history)
    loss = ((values - targets) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    optimizer.step()
    return float(loss.detach().cpu())


def resolve_target_frames(args: argparse.Namespace, current_frames: int) -> int:
    if args.total_frames is not None:
        return args.total_frames
    return current_frames + int(args.phase_frames)


def config_snapshot(args: argparse.Namespace, checkpoint: Path, target_frames: int, env: Any, pattern_sequence: list[tuple[int, ...]]) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint),
        "v1_onnx": str(Path(args.v1_onnx).expanduser()),
        "output_dir": str(output_dir(args)),
        "resume": args.resume,
        "rule": args.rule,
        "history_len": args.history_len,
        "target_frames": target_frames,
        "phase_frames": args.phase_frames,
        "pattern_sequence": [list(pattern) for pattern in pattern_sequence],
        "reward_mode": "win_loss_zero_sum",
        "training_ignores_baopei_score": True,
        "pattern_games": args.pattern_games,
        "batch_size": args.batch_size,
        "replay_size": args.replay_size,
        "train_updates_per_game": args.train_updates_per_game,
        "save_every_frames": args.save_every_frames,
        "learning_rate": args.learning_rate,
        "exp_epsilon": args.exp_epsilon,
        "cuda": args.cuda,
        "training_device": args.training_device,
        "state_shape": env.state_shape,
        "action_shape": env.action_shape,
        "history_shape": env.history_shape,
    }


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    validate_device(args)

    import torch
    from rlcard.utils import set_seed

    set_seed(args.seed)
    env = build_env(args)
    pattern_sequence = parse_pattern_sequence(args.pattern_types)
    checkpoint = selected_checkpoint(args)
    model = build_model(env, args)
    optimizers = build_optimizers(model, args)
    current_frames, stats = load_training_state(model, optimizers, checkpoint, args.training_device)
    target_frames = resolve_target_frames(args, current_frames)
    snapshot = config_snapshot(args, checkpoint, target_frames, env, pattern_sequence)
    print(json.dumps(snapshot, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return

    target_dir = output_dir(args)
    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / f"phase_{args.rule}_{target_frames}.json").open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, indent=2, ensure_ascii=False)

    v1_agent = OnnxDmcAgent(str(Path(args.v1_onnx).expanduser()))
    replays = [deque(maxlen=args.replay_size) for _ in range(PLAYER_COUNT)]
    device = "cpu" if args.training_device == "cpu" else f"cuda:{args.training_device}"
    latest_losses: dict[int, float | None] = {position: None for position in range(PLAYER_COUNT)}
    payoff_sums = [0.0 for _ in range(PLAYER_COUNT)]
    target_sums = [0.0 for _ in range(PLAYER_COUNT)]
    games = 0
    next_save_frames = current_frames + args.save_every_frames

    while current_frames < target_frames:
        games += 1
        v2_seats = set(v2_seats_for_game(games, pattern_sequence, args.pattern_games))
        agents = [
            SharedSeatAgent(model.get_agent(seat)) if seat in v2_seats else v1_agent
            for seat in range(PLAYER_COUNT)
        ]
        env.set_agents(agents)
        trajectories, payoffs = env.run(is_training=True)
        targets = win_loss_targets(payoffs)
        for seat, payoff in enumerate(payoffs):
            payoff_sums[seat] += float(payoff)
            target_sums[seat] += targets[seat]
        added = 0
        for seat in v2_seats:
            target = targets[seat]
            trajectory = trajectories[seat]
            for index in range(0, len(trajectory) - 2, 2):
                state = trajectory[index]
                action = trajectory[index + 1]
                replays[seat].append(transition_from_state(env, state, action, target))
                added += 1
        current_frames += added
        for _ in range(args.train_updates_per_game):
            for seat in v2_seats:
                loss = train_batch(
                    agent=model.get_agent(seat),
                    optimizer=optimizers[seat],
                    replay=replays[seat],
                    batch_size=args.batch_size,
                    device=device,
                    max_grad_norm=args.max_grad_norm,
                )
                if loss is not None:
                    latest_losses[seat] = loss
        if games % args.log_every_games == 0:
            avg_payoffs = [round(total / args.log_every_games, 4) for total in payoff_sums]
            avg_targets = [round(total / args.log_every_games, 4) for total in target_sums]
            replay_sizes = [len(replay) for replay in replays]
            print(
                f"games={games} frames={current_frames}/{target_frames} "
                f"v2_seats={sorted(v2_seats)} replay={replay_sizes} "
                f"losses={latest_losses} avg_payoffs={avg_payoffs} avg_targets={avg_targets}",
                flush=True,
            )
            payoff_sums = [0.0 for _ in range(PLAYER_COUNT)]
            target_sums = [0.0 for _ in range(PLAYER_COUNT)]
        if current_frames >= next_save_frames:
            path = save_training_state(model, optimizers, stats, current_frames, env, args)
            print(f"saved={path}", flush=True)
            while next_save_frames <= current_frames:
                next_save_frames += args.save_every_frames

    path = save_training_state(model, optimizers, stats, current_frames, env, args)
    print(f"finished saved={path}", flush=True)


if __name__ == "__main__":
    main()
