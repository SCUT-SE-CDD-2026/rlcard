import argparse

import numpy as np
import onnxruntime as ort
import rlcard
import torch
from rlcard.utils import get_device, set_seed


class OnnxDmcAgent:
    def __init__(self, onnx_path):
        available_providers = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.use_raw = False
        print(f"ONNX providers for {onnx_path}: {self.session.get_providers()}")

    def step(self, state):
        return self.eval_step(state)[0]

    def eval_step(self, state):
        legal_actions = state["legal_actions"]
        action_keys = list(legal_actions.keys())
        action_values = np.asarray(list(legal_actions.values()), dtype=np.float32)
        obs = np.repeat(
            np.asarray(state["obs"], dtype=np.float32)[None, :],
            len(action_keys),
            axis=0,
        )
        outputs = self.session.run(
            None,
            {
                self.input_names[0]: obs,
                self.input_names[1]: action_values,
            },
        )
        values = np.asarray(outputs[0]).reshape(-1)
        action = action_keys[int(np.argmax(values))]
        info = {"values": {k: float(v) for k, v in zip(action_keys, values)}}
        return action, info


def load_torch_agent(path, device):
    agent = torch.load(path, map_location=device, weights_only=False)
    if hasattr(agent, "set_device"):
        agent.set_device(device)
    return agent


def evaluate(env, seat0_agent, opponents, num_games):
    env.set_agents([seat0_agent, *opponents])
    sums = [0.0 for _ in range(env.num_players)]
    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        for i, payoff in enumerate(payoffs):
            sums[i] += payoff
    return [x / num_games for x in sums]


def evaluate_match(env, agents, num_games):
    env.set_agents(agents)
    sums = [0.0 for _ in range(env.num_players)]
    for _ in range(num_games):
        _, payoffs = env.run(is_training=False)
        for i, payoff in enumerate(payoffs):
            sums[i] += payoff
    return [x / num_games for x in sums]


def main():
    parser = argparse.ArgumentParser("Evaluate two ChuDaDi ONNX DMC models")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--opp1", required=True)
    parser.add_argument("--opp2", required=True)
    parser.add_argument("--opp3", required=True)
    parser.add_argument("--num-games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rule", choices=["northern", "southern"], default="northern")
    parser.add_argument(
        "--mode",
        choices=["vs_torch", "two_vs_two"],
        default="vs_torch",
        help="Compare each ONNX against fixed torch opponents, or let the two ONNX models occupy two seats each",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    env = rlcard.make(
        "chudadi", config={"seed": args.seed, "northern_rule": args.rule == "northern"}
    )

    model_a = OnnxDmcAgent(args.model_a)
    model_b = OnnxDmcAgent(args.model_b)

    print("model_a", args.model_a)
    print("model_b", args.model_b)
    print("num_games", args.num_games)
    print("rule", args.rule)

    if args.mode == "vs_torch":
        opponents = [
            load_torch_agent(args.opp1, device),
            load_torch_agent(args.opp2, device),
            load_torch_agent(args.opp3, device),
        ]
        result_a = evaluate(env, model_a, opponents, args.num_games)
        result_b = evaluate(env, model_b, opponents, args.num_games)
        print("opponents", [args.opp1, args.opp2, args.opp3])
        print("result_a", result_a)
        print("result_b", result_b)
        print("seat0_delta_b_minus_a", result_b[0] - result_a[0])
    else:
        matchup_abab = evaluate_match(
            env, [model_a, model_b, model_a, model_b], args.num_games
        )
        matchup_baba = evaluate_match(
            env, [model_b, model_a, model_b, model_a], args.num_games
        )

        model_a_avg = (
            matchup_abab[0] + matchup_abab[2] + matchup_baba[1] + matchup_baba[3]
        ) / 4.0
        model_b_avg = (
            matchup_abab[1] + matchup_abab[3] + matchup_baba[0] + matchup_baba[2]
        ) / 4.0

        print("matchup_abab", matchup_abab)
        print("matchup_baba", matchup_baba)
        print("model_a_avg", model_a_avg)
        print("model_b_avg", model_b_avg)
        print("model_b_minus_a", model_b_avg - model_a_avg)


if __name__ == "__main__":
    main()
