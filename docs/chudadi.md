# 锄大地（ChuDaDi）环境实现说明

## 概述
已根据 `docs/adding-new-environments.md` 与 `Rules_ChuDaDi.md` 完成游戏逻辑、环境封装与注册，支持在 RLCard 中通过 `rlcard.make('chudadi')` 创建环境。

## 关键规则实现
- 起手必须包含 `方块3`。
- 牌型支持单张、对子、三张、顺子、同花、葫芦（三带二）、铁支（4+1）、同花顺、炸弹。
- 顺子、同花、同花顺 **仅允许 5 张**，且不允许包含 2。
- 只能用同类型牌型压牌（炸弹可跨牌型，北方规则有条件），且点数与花色按规则比较。
- **北方规则**：有牌必出（手上有同类型且可压的牌时必须出，不能 `pass`）；炸弹仅在无同牌型可压时才可出；同花顺 > 炸弹。
- **南方规则**：可自由 `pass`；炸弹可压万物且随时打断；炸弹 > 同花顺。
- 计分包含：基本剩牌分、`大老二`（持 2 扣分）额外加分、`锄大地`奖励。

## 动作空间说明
- 使用 52 位掩码作为动作 ID，表示具体出牌组合。
- `0` 表示 `pass`。
- `num_actions = 2**52`，精确但不适配 DQN/NFSP/CFR/RandomAgent 的密集动作空间假设。
- 如需训练，建议使用 `DMC` 或 自定义策略模型。

## 状态表示（335 维）
状态向量由以下部分拼接组成：
- `current_hand`（52）：当前手牌多热，编码顺序为 `[D, C, H, S] x [3..K, A, 2]`。
- `last_action`（52）：上一手牌的多热，编码顺序同上。
- `action_type_one_hot`（10）：上一手牌型，含 `none`/`single`/`pair`/`triple`/`straight`/`flush`/`full_house`/`four_of_a_kind`/`straight_flush`/`bomb`。
- `action_length_one_hot`（14）：上一手牌张数（0..13）。
- `leader_relative_pos`（3）：领出者相对于自己的位置（下家/对家/上家），自己领出则全 0。
- `cards_left`（42）：下家/对家/上家剩牌数 one-hot（每人 14 维，0..13）。
- `history`（156）：下家/对家/上家已打出牌的多热（每人 52 维）。
- `is_leader`（1）：当前是否轮到自己领出（上一手为空）。
- `relative_pass_mask`（3）：下家/对家/上家在当前轮次是否已 `pass`。
- `is_next_warning`（1）：下家是否只剩 1 张牌。
- `is_northern_rule`（1）：当前是否使用北方规则（1=北方，0=南方）。

## 训练方法
推荐使用 `DMC` （基于动作特征），避免巨大动作空间带来的不可枚举问题。

说明：
- DMC 会使用 `ChudadiEnv.get_action_feature` 输出的 140 维动作特征（`action(52)` + `future_hand(52)` + `action_type(10)` + `action_main_rank(13)` + `action_kicker_rank(13)`）。
- 如果需要评估，可用自定义脚本调用 `env.run(is_training=False)` 或用 `tournament` 评估（`examples/evaluate.py` 同样需要把 `chudadi` 加入 `choices`）。

## 动作特征（140 维）
为了解决“打出大牌会被高估、却忽略机会成本”的问题，动作特征在 `action(52)` 基础上拼接 `future_hand(52)`，并显式加入牌型与主牌/副牌信息：
- `action(52)`：本次出牌的多热。
- `future_hand(52)`：执行该动作后，手里剩余牌的多热（即 `current_hand - action`）。
- `action_type(10)`：`none`/`single`/`pair`/`triple`/`straight`/`flush`/`full_house`/`four_of_a_kind`/`straight_flush`/`bomb` 的 one-hot。
- `action_main_rank(13)`：用于比较大小的核心点数 one-hot（如 `KKKAA` 和 `KKK44` 均为 `K`）。
- `action_kicker_rank(13)`：副牌点数 one-hot，在 `straight`/`flush`/`straight_flush` 使用次大牌点数，其它牌型按规则（如 `full_house`/`four_of_a_kind`）设置，非适用则为 0。

这样网络可以直接看到“打出 KKK+AA 会让手里失去 AA”的事实，从而更容易学到保留关键牌的长期价值，以及趁机出掉小牌的机会。

## DMC 训练流程简图
```text
state(obs=335) + action_feature(140)
            \         /
            concat(475)
                 |
               MLP x5
                 |
               Q(s,a)
```

### 使用 GPU 训练
确保当前 Python 环境中的 `torch` 支持 CUDA，然后执行：
```bash
python -m examples.run_dmc --env chudadi --cuda 0 --training_device 0 --xpid chudadi --savedir experiments/dmc_result
```

说明：
- `--cuda` 会写入 `CUDA_VISIBLE_DEVICES`，决定进程可见的 GPU 列表。
- `--training_device` 选择用于训练的 GPU 索引（相对于 `CUDA_VISIBLE_DEVICES` 的编号）。
- 若无 GPU 或 `torch.cuda.is_available()` 为 `False`，训练将回退到 CPU。

## 训练速度优化
当前 DMC 训练的瓶颈主要在 CPU 侧环境模拟（GPU 负载低很正常）。可以按以下顺序尝试：
- 提高 `--num_actors`（与 CPU 核心数接近），让更多进程并行采样。
- 适当提高 `--num_actor_devices`（通常保持 1 即可），避免过多 GPU 上下文切换。
- 设置 `OMP_NUM_THREADS=1` 和 `MKL_NUM_THREADS=1`，减少线程争用。
- 本仓库已对锄大地的动作生成做过轻量优化（减少重复排序与对象开销）。
- 已将合法动作特征改为 NumPy 向量化构建，避免逐动作 Python 循环开销。

## 与 doudizhu 的性能差异
- `doudizhu` 使用预计算动作空间与映射表（`rlcard/games/doudizhu/jsondata.*`），合法动作生成更偏查表。
- `chudadi` 仍需要在每个决策点从当前手牌枚举动作组合，CPU 侧开销更高。
- `chudadi` 为 4 人局，单局决策次数更多，导致采样吞吐下降。

示例（更高并发采样）：
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m examples.run_dmc --env chudadi --cuda 0 --training_device 0 --num_actors 6 --xpid chudadi --savedir experiments/dmc_result --load_model
```

## 训练常见问题
如果出现 `cannot pickle 'module' object`：
- 确认已使用本仓库最新代码运行，且 `rlcard/envs/chudadi.py` 中不保留对模块对象的引用（避免 multiprocessing `spawn` 序列化失败）。

## 单测
运行 chudadi 环境相关测试：
```bash
pytest -q tests/envs/test_chudadi_env.py
```

## 训练数据分析
DMC 会在 `savedir/xpid/` 下写入训练日志：
- `logs.csv`：训练曲线数据，包含 `frames`、`mean_episode_return_0..3`、`loss_0..3`。
- `fields.csv`：日志字段列表。
- `out.log`：运行日志。
- `meta.json`：参数与运行元信息。

快速查看日志字段：
```bash
head -n 5 experiments/dmc_result/chudadi/fields.csv
```

绘制平均回报曲线示例：
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('experiments/dmc_result/chudadi/logs.csv', comment='#')
for p in range(4):
    plt.plot(df['frames'], df[f'mean_episode_return_{p}'], label=f'P{p}')
plt.legend()
plt.xlabel('frames')
plt.ylabel('mean_episode_return')
plt.show()
```

## 可选后续
- 提供 DMC 训练示例脚本。
- 若需要 DQN/NFSP 训练，改造为可枚举动作空间并提供对应编码与解码方案。
