# 锄大地常用命令

本文整理当前仓库里和 `chudadi` 相关的常用命令，覆盖：

- 环境与单测
- 训练与续训
- 测试对局
- 模型评估
- 导出单文件 ONNX
- ONNX 模型对比

默认工作目录为仓库根目录：`E:\Projects\rlcard`

## 1. 环境检查

### 1.1 检查 Python / Torch / CUDA

语法：

```bash
python -c "import torch, rlcard; print(torch.__version__); print(torch.cuda.is_available()); print(rlcard.__file__)"
```

示例：

```bash
python -c "import torch, rlcard; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('rlcard', rlcard.__file__)"
```

### 1.2 检查 ONNX Runtime provider

语法：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

示例：

```bash
python -c "import onnxruntime as ort; print('providers', ort.get_available_providers())"
```

## 2. 单测与环境验证

### 2.1 运行 `chudadi` 环境测试

语法：

```bash
python -m pytest -q tests/envs/test_chudadi_env.py
```

示例：

```bash
python -m pytest -q tests/envs/test_chudadi_env.py
```

### 2.2 快速创建环境并检查状态 / 动作维度

语法：

```bash
python -c "import rlcard; env=rlcard.make('chudadi', config={...}); print(env.state_shape, env.action_shape)"
```

北方规则示例：

```bash
python -c "import rlcard; env=rlcard.make('chudadi', config={'northern_rule': True}); print(env.state_shape, env.action_shape)"
```

南方规则示例：

```bash
python -c "import rlcard; env=rlcard.make('chudadi', config={'northern_rule': False}); print(env.state_shape, env.action_shape)"
```

## 3. 训练命令

当前训练入口：`examples/run_dmc.py`

支持的关键参数：

- `--env chudadi`
- `--rule northern|southern`
- `--cuda`
- `--training_device`
- `--xpid`
- `--savedir`
- `--num_actors`
- `--num_actor_devices`
- `--save_interval`
- `--total_frames`
- `--load_model`

### 3.1 新开训练

语法：

```bash
python -m examples.run_dmc \
  --env chudadi \
  --rule <northern|southern> \
  --cuda <visible_gpu_ids> \
  --training_device <device_index> \
  --xpid <experiment_name> \
  --savedir <output_dir> \
  --num_actors <num> \
  --num_actor_devices <num> \
  --save_interval <minutes> \
  --total_frames <frames>
```

北方规则示例：

```bash
python -m examples.run_dmc \
  --env chudadi \
  --rule northern \
  --cuda 0 \
  --training_device 0 \
  --xpid chudadi_northern \
  --savedir experiments/dmc_result \
  --num_actors 1 \
  --num_actor_devices 1 \
  --save_interval 10 \
  --total_frames 200000
```

南方规则示例：

```bash
python -m examples.run_dmc \
  --env chudadi \
  --rule southern \
  --cuda 0 \
  --training_device 0 \
  --xpid chudadi_southern \
  --savedir experiments/dmc_result \
  --num_actors 1 \
  --num_actor_devices 1 \
  --save_interval 10 \
  --total_frames 200000
```

### 3.2 在现有模型基础上续训

语法：

```bash
python -m examples.run_dmc \
  --env chudadi \
  --rule <northern|southern> \
  --cuda <visible_gpu_ids> \
  --training_device <device_index> \
  --xpid <existing_experiment_name> \
  --savedir <output_dir> \
  --load_model \
  --total_frames <target_total_frames>
```

说明：

- `--load_model` 表示从当前 `xpid` 目录下的 `model.tar` 继续训练
- `--total_frames` 是目标总帧数，不是“再加多少帧”

示例：从 `70400` 左右继续跑到 `170400`：

```bash
python -m examples.run_dmc \
  --env chudadi \
  --rule northern \
  --cuda 0 \
  --training_device 0 \
  --xpid chudadi \
  --savedir experiments/dmc_result \
  --num_actors 1 \
  --num_actor_devices 1 \
  --load_model \
  --total_frames 170400
```

### 3.3 训练输出目录说明

以 `xpid=chudadi` 为例，输出目录通常为：

```text
experiments/dmc_result/chudadi/
```

常见文件：

- `model.tar`：续训状态文件
- `0_<frames>.pth` ~ `3_<frames>.pth`：4 个 seat 的模型
- `logs.csv`：训练日志
- `fields.csv`：日志字段
- `meta.json`：运行参数
- `out.log`：控制台日志

## 4. 测试对局

当前测试对局脚本：`play_game.py`

### 4.1 随机对局

语法：

```bash
python play_game.py --env chudadi --num_games <num>
```

示例：

```bash
python play_game.py --env chudadi --num_games 1
```

### 4.2 指定某个 seat 使用训练好的 `.pth` 模型

语法：

```bash
python play_game.py \
  --env chudadi \
  --num_games <num> \
  --model_path <checkpoint.pth> \
  --model_position <seat_id> \
  --unsafe_torch_load
```

示例：

```bash
python play_game.py \
  --env chudadi \
  --num_games 1 \
  --model_path experiments/dmc_result/chudadi/0_220800.pth \
  --model_position 0 \
  --unsafe_torch_load
```

说明：

- 其他 seat 默认仍然是 `RandomAgent`
- Windows 终端如果遇到花色 Unicode 输出问题，可先切 UTF-8 或临时修改终端编码

## 5. 导出单文件 ONNX

当前导出脚本：`docs/onnx_android/export_onnx.py`

### 5.1 导出单文件 ONNX

语法：

```bash
python docs/onnx_android/export_onnx.py \
  --checkpoint <checkpoint.pth> \
  --onnx <output.onnx> \
  --rlcard-dmc \
  --obs-shape 1,335 \
  --action-shape 1,140 \
  --obs-name obs \
  --action-name actions \
  --output-name value \
  --dynamic-batch \
  --single-file
```

示例：

```bash
$env:PYTHONIOENCODING='utf-8'; python docs/onnx_android/export_onnx.py \
  --checkpoint experiments/dmc_result/chudadi/0_220800.pth \
  --onnx experiments/dmc_result/chudadi/onnx/0_220800.onnx \
  --rlcard-dmc \
  --obs-shape 1,335 \
  --action-shape 1,140 \
  --obs-name obs \
  --action-name actions \
  --output-name value \
  --dynamic-batch \
  --single-file
```

说明：

- `--single-file` 会尽量把权重嵌进单个 `.onnx`
- 不加 `--single-file` 时，PyTorch 新导出器可能会生成 `.onnx + .onnx.data`
- Windows 下建议设置：`$env:PYTHONIOENCODING='utf-8'`

### 5.2 校验 ONNX

语法：

```bash
python docs/onnx_android/check_onnx.py \
  --onnx <model.onnx> \
  --input-shape 1,335 1,140 \
  --input-name obs actions
```

示例：

```bash
$env:PYTHONIOENCODING='utf-8'; python docs/onnx_android/check_onnx.py \
  --onnx "E:\Projects\rlcard\experiments\dmc_result\chudadi\onnx\0_220800.onnx" \
  --input-shape 1,335 1,140 \
  --input-name obs actions
```

### 5.3 直接用 Python 校验单文件 ONNX

当 `check_onnx.py` 在 Windows 下遇到路径或编码问题时，可直接运行：

```bash
python -c "import onnx, onnxruntime as ort, numpy as np; p=r'E:\Projects\rlcard\experiments\dmc_result\chudadi\onnx\0_220800.onnx'; m=onnx.load(p); onnx.checker.check_model(m); sess=ort.InferenceSession(p, providers=['CPUExecutionProvider']); out=sess.run(None, {'obs': np.random.randn(1,335).astype(np.float32), 'actions': np.random.randn(1,140).astype(np.float32)}); print(out[0].shape)"
```

## 6. 新旧 checkpoint 强度对比

当前脚本：`tools/eval_chudadi_checkpoints.py`

用途：

- 对比新旧 `.pth` 模型在 seat0 的表现
- 对手固定为旧版 seat1/2/3

语法：

```bash
python tools/eval_chudadi_checkpoints.py \
  --old <old_seat0.pth> \
  --new <new_seat0.pth> \
  --opp1 <seat1_old.pth> \
  --opp2 <seat2_old.pth> \
  --opp3 <seat3_old.pth> \
  --num_games <num>
```

示例：

```bash
python tools/eval_chudadi_checkpoints.py \
  --old experiments/dmc_result/chudadi/0_70400.pth \
  --new experiments/dmc_result/chudadi/0_220800.pth \
  --opp1 experiments/dmc_result/chudadi/1_70400.pth \
  --opp2 experiments/dmc_result/chudadi/2_70400.pth \
  --opp3 experiments/dmc_result/chudadi/3_70400.pth \
  --num_games 200
```

## 7. 两个 ONNX 模型互打对比

当前脚本：`tools/eval_chudadi_onnx.py`

### 7.1 两个 ONNX 各占两个席位互打

用途：

- `model_a` 占两个 seat
- `model_b` 占两个 seat
- 自动跑两种排位：`A B A B` 和 `B A B A`
- 输出两个模型的平均 payoff

语法：

```bash
python tools/eval_chudadi_onnx.py \
  --mode two_vs_two \
  --model-a <model_a.onnx> \
  --model-b <model_b.onnx> \
  --opp1 <placeholder> \
  --opp2 <placeholder> \
  --opp3 <placeholder> \
  --num-games <num> \
  --rule <northern|southern>
```

说明：

- 当前脚本为兼容旧参数，`--opp1/--opp2/--opp3` 在 `two_vs_two` 模式下不会实际使用

示例：

```bash
python tools/eval_chudadi_onnx.py \
  --mode two_vs_two \
  --model-a experiments/dmc_result/chudadi/onnx/0_220800.onnx \
  --model-b experiments/dmc_result/chudadi/onnx/wsytest.onnx \
  --opp1 experiments/dmc_result/chudadi/1_70400.pth \
  --opp2 experiments/dmc_result/chudadi/2_70400.pth \
  --opp3 experiments/dmc_result/chudadi/3_70400.pth \
  --num-games 500 \
  --rule northern
```

### 7.2 单个 ONNX 对固定旧版 `.pth` 对手

语法：

```bash
python tools/eval_chudadi_onnx.py \
  --mode vs_torch \
  --model-a <model_a.onnx> \
  --model-b <model_b.onnx> \
  --opp1 <seat1_old.pth> \
  --opp2 <seat2_old.pth> \
  --opp3 <seat3_old.pth> \
  --num-games <num> \
  --rule <northern|southern>
```

## 8. 常见注意事项

### 8.1 训练默认不会自动很快结束

- 训练停止依据是 `--total_frames`
- 不传时默认是一个非常大的值
- 建议显式指定 `--total_frames`

### 8.2 `--load_model` 的含义

- 表示从当前 `xpid` 目录下 `model.tar` 继续训练
- 它不是自动“再训 100000 frames”
- 你应当同时指定新的目标 `--total_frames`

### 8.3 ONNX Runtime 是否真的在用 GPU

检查命令：

```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

如果没有 `CUDAExecutionProvider`，则 ONNX 推理会回退到 CPU。

### 8.4 Windows 终端编码

某些导出和测试对局脚本在 Windows `gbk` 控制台下可能遇到 Unicode 输出问题。

建议先设置：

```powershell
$env:PYTHONIOENCODING='utf-8'
```

再执行导出或校验命令。
