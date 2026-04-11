import argparse
from typing import List

import numpy as np
import onnx
import onnxruntime as ort


def parse_shape(shape: str) -> List[int]:
    try:
        return [int(x) for x in shape.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"Invalid --input-shape: {shape}") from exc


def resolve_dtype(dtype: str):
    mapping = {
        "float32": np.float32,
        "float16": np.float16,
        "int64": np.int64,
        "int32": np.int32,
    }
    if dtype not in mapping:
        raise SystemExit(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ONNX and run a test inference")
    parser.add_argument("--onnx", required=True, help="ONNX model path")
    parser.add_argument(
        "--input-shape",
        required=True,
        nargs="+",
        help="e.g. 1,3,224,224 (repeat for each model input)",
    )
    parser.add_argument(
        "--input-name",
        nargs="+",
        help="Override input name(s) (repeat for each model input)",
    )
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)
    print("ONNX check OK")

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    input_shapes = args.input_shape
    input_names = args.input_name or [inp.name for inp in inputs]

    if len(input_shapes) != len(inputs):
        raise SystemExit(
            f"Model has {len(inputs)} input(s), but {len(input_shapes)} shape(s) provided. "
            f"Expected shapes for: {[inp.name for inp in inputs]}"
        )
    if len(input_names) != len(inputs):
        raise SystemExit(
            f"Model has {len(inputs)} input(s), but {len(input_names)} name(s) provided."
        )

    dtype = resolve_dtype(args.dtype)
    feed = {}
    for _, shape_str, name in zip(inputs, input_shapes, input_names):
        shape = parse_shape(shape_str)
        dummy = (
            np.random.randn(*shape).astype(dtype)
            if np.issubdtype(dtype, np.floating)
            else np.zeros(shape, dtype=dtype)
        )
        feed[name] = dummy
        print(f"Input '{name}' shape: {shape}, dtype: {dtype}")

    outputs = sess.run(None, feed)
    print("Run OK")
    if outputs:
        for idx, out in enumerate(outputs):
            print(f"Output[{idx}] shape:", out.shape)


if __name__ == "__main__":
    main()
