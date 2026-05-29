"""An example of training a Deep Monte-Carlo (DMC) Agent on the environments in RLCard"""

import os
import argparse

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer


def train(args):
    config = {}
    model_version = args.model_version
    if args.env == "chudadi":
        config["northern_rule"] = args.rule == "northern"
        config["history_len"] = args.history_len
        if model_version == "auto":
            model_version = "v2"
    elif model_version == "auto":
        model_version = "v1"
    env = rlcard.make(args.env, config=config)

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
        total_frames=args.total_frames,
        batch_size=args.batch_size,
        unroll_length=args.unroll_length,
        num_buffers=args.num_buffers,
        num_threads=args.num_threads,
        model_version=model_version,
        history_shape=getattr(env, "history_shape", None) if model_version == "v2" else None,
    )
    trainer.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DMC example in RLCard")
    parser.add_argument(
        "--env",
        type=str,
        default="leduc-holdem",
        choices=[
            "blackjack",
            "leduc-holdem",
            "limit-holdem",
            "doudizhu",
            "mahjong",
            "no-limit-holdem",
            "uno",
            "gin-rummy",
            "chudadi",
        ],
    )
    parser.add_argument("--cuda", type=str, default="")
    parser.add_argument("--load_model", action="store_true", help="Load an existing model")
    parser.add_argument("--xpid", default="leduc_holdem", help="Experiment id")
    parser.add_argument("--savedir", default="experiments/dmc_result", help="Root dir for results")
    parser.add_argument("--save_interval", default=30, type=int, help="Checkpoint interval in minutes")
    parser.add_argument("--num_actor_devices", default=1, type=int)
    parser.add_argument("--num_actors", default=5, type=int)
    parser.add_argument("--training_device", default="0", type=str)
    parser.add_argument("--total_frames", default=100000000000, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--unroll_length", default=100, type=int)
    parser.add_argument("--num_buffers", default=50, type=int)
    parser.add_argument("--num_threads", default=4, type=int)
    parser.add_argument(
        "--rule",
        default="northern",
        choices=["northern", "southern"],
        help="ChuDaDi rule set to use during training",
    )
    parser.add_argument(
        "--model-version",
        default="auto",
        choices=["auto", "v1", "v2"],
        help="Use V2 by default for ChuDaDi; V1 elsewhere.",
    )
    parser.add_argument("--history-len", default=13, type=int)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
