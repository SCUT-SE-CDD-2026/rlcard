import argparse
import copy
import json
import re
from pathlib import Path

import rlcard
import torch
from rlcard.agents.dmc_agent.model import DMCAgent, DMCAgentV2
from rlcard.utils import set_seed


class SharedSeatAgent:
    def __init__(self, agent):
        self.agent = agent
        self.use_raw = False

    def step(self, state):
        return self.agent.step(state)

    def eval_step(self, state):
        return self.agent.eval_step(state)


def parse_args():
    parser = argparse.ArgumentParser(
        "Train one ChuDaDi V2 DMC agent while alternating northern/southern rules by fixed game blocks."
    )
    parser.add_argument("--checkpoint", help="Existing V2 .pth checkpoint to continue from; omit to start fresh")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--xpid", default="chudadi_v2_gru_mixed_alt")
    parser.add_argument("--total-games", type=int, default=100000)
    parser.add_argument("--switch-games", type=int, default=120, help="Rule switch period in completed games")
    parser.add_argument("--history-len", type=int, default=13)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--exploration", type=float, default=0.01)
    parser.add_argument("--save-every-games", type=int, default=1000)
    parser.add_argument("--log-every-games", type=int, default=120)
    parser.add_argument("--max-grad-norm", type=float, default=40.0)
    parser.add_argument("--train-updates-per-game", type=int, default=2)
    parser.add_argument("--opponent-refresh-games", type=int, default=120)
    parser.add_argument("--warmup-games", type=int, default=20)
    return parser.parse_args()


def set_agent_device(agent, device):
    agent.device = device
    agent.net.to(device)


def load_agent(path, device, exploration):
    agent = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(agent, (DMCAgent, DMCAgentV2)):
        raise TypeError(f"Unsupported checkpoint type: {type(agent)!r}")
    if getattr(agent, "model_version", None) != "v2":
        raise ValueError("This script requires a V2 DMCAgent checkpoint")
    set_agent_device(agent, device)
    agent.exp_epsilon = exploration
    return agent


def create_agent(env, device, exploration):
    agent = DMCAgent(
        state_shape=env.state_shape[0],
        action_shape=env.action_shape[0],
        exp_epsilon=exploration,
        device="cpu",
        model_version="v2",
        history_shape=env.history_shape[0],
    )
    set_agent_device(agent, device)
    return agent


def clone_eval_agent(agent, device):
    cloned = copy.deepcopy(agent)
    set_agent_device(cloned, device)
    cloned.exp_epsilon = agent.exp_epsilon
    cloned.eval()
    return cloned


def checkpoint_frame_from_name(path):
    match = re.search(r"_(\d+)\.pth$", Path(path).name)
    return int(match.group(1)) if match else 0


def save_agent(agent, output_dir, games, source_frame):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"1_{source_frame + games}.pth"
    torch.save(agent, path)
    latest = output_dir / "latest.pth"
    torch.save(agent, latest)
    return path


def collect_transitions(env, agents, train_agent):
    env.set_agents(agents)
    trajectories, payoffs = env.run(is_training=True)
    transitions = []
    for player_id, trajectory in enumerate(trajectories):
        payoff = float(payoffs[player_id])
        for index in range(0, len(trajectory) - 2, 2):
            state = trajectory[index]
            action = trajectory[index + 1]
            transitions.append(
                (
                    torch.from_numpy(state["obs"]).float(),
                    torch.from_numpy(env.get_action_feature(action, state)).float(),
                    torch.from_numpy(state["history"]).float(),
                    payoff,
                )
            )
    return transitions, payoffs


def train_batch(agent, optimizer, replay, batch_size, device, max_grad_norm):
    if len(replay) < batch_size:
        return None
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


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    output_dir = Path(args.output_dir) / args.xpid
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2, ensure_ascii=False)

    northern_env = rlcard.make(
        "chudadi",
        config={"seed": args.seed, "northern_rule": True, "history_len": args.history_len},
    )
    southern_env = rlcard.make(
        "chudadi",
        config={"seed": args.seed + 1, "northern_rule": False, "history_len": args.history_len},
    )

    if args.checkpoint:
        agent = load_agent(args.checkpoint, device, args.exploration)
        source_frame = checkpoint_frame_from_name(args.checkpoint)
    else:
        agent = create_agent(northern_env, device, args.exploration)
        source_frame = 0
    optimizer = torch.optim.RMSprop(agent.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-5)
    replay = []
    latest_loss = None
    payoff_sums = [0.0, 0.0, 0.0, 0.0]
    rule_counts = {"northern": 0, "southern": 0}

    print(f"device={device}")
    print(f"checkpoint={args.checkpoint or '<fresh enhanced V2>'}")
    print(f"output_dir={output_dir}")
    print(f"switch_games={args.switch_games}")

    frozen = clone_eval_agent(agent, device)

    for game_index in range(1, args.total_games + 1):
        use_northern = ((game_index - 1) // args.switch_games) % 2 == 0
        rule_name = "northern" if use_northern else "southern"
        env = northern_env if use_northern else southern_env
        if game_index == 1 or (game_index - 1) % args.opponent_refresh_games == 0:
            frozen = clone_eval_agent(agent, device)
        agents = [SharedSeatAgent(agent)] + [SharedSeatAgent(frozen) for _ in range(3)]
        transitions, payoffs = collect_transitions(env, agents, agent)
        replay.extend(transitions)
        if len(replay) > args.replay_size:
            del replay[: len(replay) - args.replay_size]
        if game_index >= args.warmup_games:
            for _ in range(args.train_updates_per_game):
                loss = train_batch(agent, optimizer, replay, args.batch_size, device, args.max_grad_norm)
                if loss is not None:
                    latest_loss = loss
        for seat, payoff in enumerate(payoffs):
            payoff_sums[seat] += float(payoff)
        rule_counts[rule_name] += 1

        if game_index % args.log_every_games == 0:
            divisor = max(1, sum(rule_counts.values()))
            averages = [round(total / divisor, 4) for total in payoff_sums]
            print(
                f"games={game_index} rules={rule_counts} replay={len(replay)} "
                f"loss={latest_loss} avg_payoffs={averages}",
                flush=True,
            )
            payoff_sums = [0.0, 0.0, 0.0, 0.0]
            rule_counts = {"northern": 0, "southern": 0}

        if game_index % args.save_every_games == 0:
            path = save_agent(agent, output_dir, game_index, source_frame)
            print(f"saved={path}", flush=True)

    path = save_agent(agent, output_dir, args.total_games, source_frame)
    print(f"finished saved={path}")


if __name__ == "__main__":
    main()
