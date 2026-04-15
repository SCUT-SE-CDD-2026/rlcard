import argparse

import rlcard
import torch
from rlcard.utils import get_device


def load_agent(path, device):
    agent = torch.load(path, map_location=device, weights_only=False)
    if hasattr(agent, "set_device"):
        agent.set_device(device)
    return agent


def evaluate(env, agents, num_games):
    env.set_agents(agents)
    payoff_sums = [0.0 for _ in range(env.num_players)]
    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        for i, payoff in enumerate(payoffs):
            payoff_sums[i] += payoff
    return [x / num_games for x in payoff_sums]


def main():
    parser = argparse.ArgumentParser("Evaluate old vs new chudadi checkpoints")
    parser.add_argument("--old", required=True, help="Old seat-0 checkpoint")
    parser.add_argument("--new", required=True, help="New seat-0 checkpoint")
    parser.add_argument("--opp1", required=True, help="Seat-1 opponent checkpoint")
    parser.add_argument("--opp2", required=True, help="Seat-2 opponent checkpoint")
    parser.add_argument("--opp3", required=True, help="Seat-3 opponent checkpoint")
    parser.add_argument("--num_games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    env = rlcard.make("chudadi", config={"seed": args.seed})

    old_agent = load_agent(args.old, device)
    new_agent = load_agent(args.new, device)
    opp1 = load_agent(args.opp1, device)
    opp2 = load_agent(args.opp2, device)
    opp3 = load_agent(args.opp3, device)

    old_result = evaluate(env, [old_agent, opp1, opp2, opp3], args.num_games)
    new_result = evaluate(env, [new_agent, opp1, opp2, opp3], args.num_games)

    print("old_vs_old_opponents", old_result)
    print("new_vs_old_opponents", new_result)
    print("seat0_delta", new_result[0] - old_result[0])


if __name__ == "__main__":
    main()
