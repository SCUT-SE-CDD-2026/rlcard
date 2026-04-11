# ONNX Runtime 在 Android Kotlin 上的打包与运行（从 .pth 导出）

本文给出从 `.pth` 导出 ONNX、验证模型、在 Android Kotlin 上集成 ONNX Runtime 并运行推理的**完整流程与命令**，并提供可直接改造的脚本文件。

## 目录与脚本

* `docs/onnx_android/export_onnx.py`
* `docs/onnx_android/check_onnx.py`

## 1. 前置条件

* 具备 Python 环境与 PyTorch（用于导出）
* 安装依赖：

```bash
pip install torch onnx onnxruntime
```

* Android Studio / Gradle 项目（Kotlin）

## 2. 从 `.pth` 导出 ONNX

推荐使用脚本：`docs/onnx_android/export_onnx.py`。

* **方式 A**：在脚本中修改 `build_model()`（最直接，适合模型初始化需要参数时）
* **方式 B**：通过 `--model-module/--model-class` 传入模型类（需要无参构造）
* **方式 C**：直接加载 `.pth` 里保存的完整模型（`nn.Module` 实例），无需提供类信息
* **方式 D**：RLCard DMC（如 ChuDaDi），直接加载 `.pth` 里保存的 `DMCAgent` 并导出双输入 ONNX

输出 ONNX 的上级目录不存在时会**自动创建**。

### 2.1 RLCard ChuDaDi DMC 导出示例

```bash
python docs/onnx_android/export_onnx.py \
  --checkpoint experiments/dmc_result/chudadi/3_46003200.pth \
  --onnx experiments/dmc_result/chudadi/onnx/3_46003200.onnx \
  --rlcard-dmc \
  --obs-shape 1,335 \
  --action-shape 1,140 \
  --obs-name obs \
  --action-name actions \
  --output-name value \
  --dynamic-batch
```

### 2.2 脚本核心导出逻辑

与命令行参数一致的核心逻辑如下：

```python
import torch

model = MyModel(...)  # 或从 checkpoint 直接加载完整模型
obs = torch.randn(1, 335)
actions = torch.randn(1, 140)

torch.onnx.export(
    model,
    (obs, actions),
    "model.onnx",
    opset_version=17,
    input_names=["obs", "actions"],
    output_names=["value"],
    dynamic_axes={
        "obs": {0: "batch"},
        "actions": {0: "batch"},
        "value": {0: "batch"},
    },
    do_constant_folding=True,
)
print("Exported model.onnx")
```

### 2.3 注意事项

* `opset_version` 按需调整，Android 端 ONNX Runtime 必须支持该 opset。
* `input_names` / `output_names` 必须与 Android 推理代码一致。
* `dummy_input` 的形状与类型要与真实输入一致（含预处理后的 dtype）。
* RLCard DMC 模型为双输入：`obs`（335 维）和 `actions`（140 维），输出为 `value`。

### 2.4 `export_onnx.py` 参数说明

| 参数 | 说明 |
|------|------|
| `--checkpoint` | `.pth` 路径 |
| `--onnx` | 输出 ONNX 路径（上级目录自动创建） |
| `--input-shape` | 输入形状（通用模型使用，如 `1,3,224,224`） |
| `--opset` | opset 版本，默认 `17` |
| `--input-name` / `--output-name` | 输入/输出名 |
| `--dynamic-batch` | 将 batch 维设为动态 |
| `--dtype` | 输入 dtype（`float32/float16/int64/int32`） |
| `--device` | `cpu` 或 `cuda` |
| `--model-module` / `--model-class` | 从模块加载模型类（可选） |
| `--rlcard-dmc` | 加载 RLCard DMC agent `.pth`，导出双输入模型 |
| `--obs-shape` | RLCard DMC 的 obs 形状，默认 `1,335`（ChuDaDi） |
| `--action-shape` | RLCard DMC 的 actions 形状，默认 `1,140`（ChuDaDi） |
| `--obs-name` / `--action-name` | RLCard DMC 的输入名，默认 `obs` / `actions` |

## 3. 验证 ONNX

推荐使用脚本：`docs/onnx_android/check_onnx.py`。

### 3.1 ChuDaDi 多输入验证

```bash
python docs/onnx_android/check_onnx.py \
  --onnx experiments/dmc_result/chudadi/onnx/3_46003200.onnx \
  --input-shape 1,335 1,140 \
  --input-name obs actions
```

### 3.2 脚本核心逻辑

```python
import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX check OK")

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_names = [inp.name for inp in sess.get_inputs()]

feed = {
    "obs": np.random.randn(1, 335).astype(np.float32),
    "actions": np.random.randn(1, 140).astype(np.float32),
}
outputs = sess.run(None, feed)
print("Run OK, output[0] shape:", outputs[0].shape)
```

### 3.3 `check_onnx.py` 参数说明

| 参数 | 说明 |
|------|------|
| `--onnx` | ONNX 模型路径 |
| `--input-shape` | 输入形状（多输入时按顺序写多个，如 `1,335 1,140`） |
| `--input-name` | 覆盖输入名（多输入时按顺序写多个） |
| `--dtype` | 输入 dtype（`float32/float16/int64/int32`） |

## 4. Android Kotlin 集成

### 4.1 添加依赖（Maven AAR）

`app/build.gradle.kts`：

```kotlin
dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.x.y")
}
```

建议到 Maven Central 查询最新版本号替换 `1.x.y`。

### 4.2 放置模型与避免压缩

放置模型文件：

* `app/src/main/assets/model.onnx`

避免 AAPT 压缩（防止加载失败或耗时解压），在 AGP 7.x 及以下：

```kotlin
android {
    aaptOptions {
        noCompress += "onnx"
    }
}
```

在 AGP 8.x 请使用 `androidResources { noCompress += listOf("onnx") }`。

### 4.3 Kotlin 推理示例

```kotlin
import ai.onnxruntime.*
import android.content.Context
import java.io.File

fun runOnnx(context: Context) {
    val env = OrtEnvironment.getEnvironment()
    val opts = OrtSession.SessionOptions()

    val modelFile = File(context.cacheDir, "model.onnx")
    if (!modelFile.exists()) {
        context.assets.open("model.onnx").use { input ->
            modelFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }
    }

    val session = env.createSession(modelFile.absolutePath, opts)

    val obsData = FloatArray(1 * 335) { 0.0f }
    val obsShape = longArrayOf(1, 335)
    val obsTensor = OnnxTensor.createTensor(env, obsData, obsShape)

    val actionsData = FloatArray(1 * 140) { 0.0f }
    val actionsShape = longArrayOf(1, 140)
    val actionsTensor = OnnxTensor.createTensor(env, actionsData, actionsShape)

    val results = session.run(mapOf("obs" to obsTensor, "actions" to actionsTensor))
    val output = results[0].value

    obsTensor.close()
    actionsTensor.close()
    results.close()
    session.close()
}
```

多输入/多输出时：
* `session.inputNames` 返回所有输入名，需构造 `Map<String, OnnxTensor>`。
* `results` 返回与输出名顺序一致的列表，或用 `session.outputNames` 对齐。

### 4.4 预处理/后处理一致性

* Android 侧的输入必须与 PyTorch 训练/导出时预处理完全一致（通道顺序、归一化、dtype）。
* 若 PyTorch 端使用 `float32`，Android 侧输入也必须是 `float`。

## 5. 构建与运行

```bash
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

## 6. 常见问题

| 问题 | 原因/解决 |
|------|----------|
| 输入名不一致 | 导出时的 `input_names` 需要和 Android 代码一致。 |
| 形状不一致 | 动态 batch 只允许维度变化，其他维度必须匹配。 |
| opset 不兼容 | 导出 opset 过高会导致 Android 端加载失败。 |
