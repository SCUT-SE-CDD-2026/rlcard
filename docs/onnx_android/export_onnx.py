import argparse
import importlib
import sys
from pathlib import Path
from typing import List

import torch


def parse_shape(shape: str) -> List[int]:
    try:
        return [int(x) for x in shape.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --input-shape: {shape}") from exc


def resolve_dtype(dtype: str):
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "int64": torch.int64,
        "int32": torch.int32,
    }
    if dtype not in mapping:
        raise SystemExit(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def build_model():
    # TODO: Replace this with your model creation logic.
    # Example:
    # from my_model import MyModel
    # return MyModel(...)
    raise SystemExit(
        "Please implement build_model(), or pass --model-module/--model-class."
    )


def _add_repo_root_to_syspath() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def load_rlcard_dmc_net(checkpoint: str, device: str) -> torch.nn.Module:
    _add_repo_root_to_syspath()
    from rlcard.agents.dmc_agent.model import DMCAgent, DMCNet

    # PyTorch 2.6+ needs explicit allowlist when weights_only=True by default.
    try:
        torch.serialization.add_safe_globals([DMCAgent, DMCNet])
    except Exception:
        pass

    agent = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(agent, DMCAgent):
        agent.eval()
        return agent.net

    raise SystemExit(
        "Unsupported checkpoint for --rlcard-dmc. "
        "Please pass a per-player .pth saved by RLCard DMC."
    )


def load_model(args) -> torch.nn.Module:
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if isinstance(state, torch.nn.Module):
        # checkpoint saves the full model instead of state_dict
        state.eval()
        return state

    if args.model_module and args.model_class:
        module = importlib.import_module(args.model_module)
        model_cls = getattr(module, args.model_class)
        model = model_cls()
    else:
        model = build_model()

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export .pth to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth")
    parser.add_argument("--onnx", required=True, help="Output ONNX path")
    parser.add_argument("--input-shape", help="e.g. 1,3,224,224")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--model-module", help="Python module for model class")
    parser.add_argument("--model-class", help="Model class name")
    parser.add_argument(
        "--rlcard-dmc",
        action="store_true",
        help="Load RLCard DMC agent .pth and export its net with obs/actions inputs",
    )
    parser.add_argument("--obs-shape", default="1,335", help="e.g. 1,335 for ChuDaDi")
    parser.add_argument(
        "--action-shape", default="1,140", help="e.g. 1,140 for ChuDaDi"
    )
    parser.add_argument("--obs-name", default="obs")
    parser.add_argument("--action-name", default="actions")
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Embed weights into a single .onnx file instead of writing external .onnx.data",
    )
    args = parser.parse_args()

    dtype = resolve_dtype(args.dtype)

    Path(args.onnx).parent.mkdir(parents=True, exist_ok=True)

    if args.rlcard_dmc:
        model = load_rlcard_dmc_net(args.checkpoint, args.device).to(args.device)
        obs_shape = parse_shape(args.obs_shape)
        action_shape = parse_shape(args.action_shape)
        if dtype.is_floating_point:
            obs = torch.randn(*obs_shape, device=args.device, dtype=dtype)
            actions = torch.randn(*action_shape, device=args.device, dtype=dtype)
        else:
            obs = torch.zeros(*obs_shape, device=args.device, dtype=dtype)
            actions = torch.zeros(*action_shape, device=args.device, dtype=dtype)

        dynamic_axes = None
        if args.dynamic_batch:
            dynamic_axes = {
                args.obs_name: {0: "batch"},
                args.action_name: {0: "batch"},
                args.output_name: {0: "batch"},
            }

        torch.onnx.export(
            model,
            (obs, actions),
            args.onnx,
            opset_version=args.opset,
            input_names=[args.obs_name, args.action_name],
            output_names=[args.output_name],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            external_data=not args.single_file,
        )
    else:
        if not args.input_shape:
            raise SystemExit("--input-shape is required unless --rlcard-dmc is set.")
        model = load_model(args).to(args.device)
        shape = parse_shape(args.input_shape)
        if dtype.is_floating_point:
            dummy = torch.randn(*shape, device=args.device, dtype=dtype)
        else:
            dummy = torch.zeros(*shape, device=args.device, dtype=dtype)

        dynamic_axes = None
        if args.dynamic_batch:
            dynamic_axes = {
                args.input_name: {0: "batch"},
                args.output_name: {0: "batch"},
            }

        torch.onnx.export(
            model,
            dummy,
            args.onnx,
            opset_version=args.opset,
            input_names=[args.input_name],
            output_names=[args.output_name],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            external_data=not args.single_file,
        )
    print(f"Exported {args.onnx}")


if __name__ == "__main__":
    main()
