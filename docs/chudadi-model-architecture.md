# 锄大地模型架构与训练数据流

## 概览

当前仓库中的“锄大地”模型不是直接输出整张动作概率表的策略网络，而是一个基于候选动作逐个打分的 `Deep Monte-Carlo (DMC)` 价值网络。

它的基本形式是：

```text
(state, candidate_action) -> scalar value
```

在推理阶段：

- 枚举当前所有合法动作
- 为每个合法动作构造动作特征
- 将同一个状态与每个候选动作拼接后送入网络
- 得到每个候选动作的分数
- 选择分数最高的动作

在训练阶段：

- 先完整打完一局
- 对某个玩家这一整局中出现过的每个决策点 `(s, a)`
- 都使用该玩家最终的对局 payoff 作为监督信号
- 用均方误差回归网络输出

这意味着当前实现更接近“Monte Carlo 回报监督下的动作价值回归”，而不是 DQN 风格的时序差分学习。

## 关键代码位置

- 环境状态与动作编码：`rlcard/envs/chudadi.py`
- 游戏流程与 payoff：`rlcard/games/chudadi/game.py`
- 合法动作与牌型规则：`rlcard/games/chudadi/utils.py`
- 对局计分：`rlcard/games/chudadi/judger.py`
- DMC 网络结构：`rlcard/agents/dmc_agent/model.py`
- DMC 采样与 buffer 写入：`rlcard/agents/dmc_agent/utils.py`
- DMC 训练循环：`rlcard/agents/dmc_agent/trainer.py`
- 启动训练入口：`examples/run_dmc.py`

## 整体结构

训练和推理可以概括为下面这条链路：

```text
ChuDaDiEnv
  -> 生成 obs(335) 和 legal_actions{action_id: feature(140)}
  -> DMCAgent 对每个合法动作分别打分
  -> 选择动作并推进对局
  -> 一局结束后得到每个玩家的 payoff
  -> 用该 payoff 监督本局中该玩家所有 (s, a) 样本
  -> DMCNet 更新参数
```

对于 `chudadi`：

- 玩家数：4
- 每个座位一个独立模型
- 状态维度：`335`
- 动作特征维度：`140`
- 网络输入维度：`335 + 140 = 475`

## 端到端 ASCII 流程图

下面这张图把环境编码、推理、采样、训练和权重同步串了起来：

```text
                        +----------------------+
                        |   ChuDaDi Game Core  |
                        | game/round/judger    |
                        +----------+-----------+
                                   |
                                   | raw state
                                   v
                        +----------------------+
                        |    ChudadiEnv        |
                        | _extract_state()     |
                        +----------+-----------+
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
           obs: [335]                   legal actions: {id -> [140]}
                    \                           /
                     \                         /
                      \     repeat + stack    /
                       +---------------------+
                       | DMCAgent.predict()  |
                       +----------+----------+
                                  |
                                  | obs [K,335], act [K,140]
                                  v
                       +---------------------+
                       |       DMCNet        |
                       |   concat -> MLP     |
                       +----------+----------+
                                  |
                                  | values [K]
                                  v
                       +---------------------+
                       | argmax / epsilon    |
                       | choose action id    |
                       +----------+----------+
                                  |
                                  | step(action)
                                  v
                        +----------------------+
                        |   Env.run() episode   |
                        | trajectories, payoff  |
                        +----------+-----------+
                                   |
                                   | for each (s,a) of player p
                                   | target = final payoff[p]
                                   v
                        +----------------------+
                        |  Shared Buffers      |
                        | state[T,335]         |
                        | action[T,140]        |
                        | target[T]            |
                        +----------+-----------+
                                   |
                                   | stack B buffers
                                   v
                        +----------------------+
                        |   Learner Batch      |
                        | state[T,B,335]       |
                        | act[T,B,140]         |
                        | target[T,B]          |
                        +----------+-----------+
                                   |
                                   | flatten -> [T*B, ...]
                                   v
                        +----------------------+
                        | learn() + MSE loss   |
                        | (Q(s,a)-G)^2         |
                        +----------+-----------+
                                   |
                                   | optimizer.step()
                                   v
                        +----------------------+
                        | updated seat model   |
                        +----------+-----------+
                                   |
                                   | sync weights
                                   v
                        +----------------------+
                        | actor model copies   |
                        +----------------------+
```

## 状态表示：335 维

`ChudadiEnv._extract_state()` 会把原始游戏状态编码成一个 `obs` 向量。

当前维度组成如下：

- `current_hand`: `52`
- `last_action`: `52`
- `action_type_one_hot`: `10`
- `action_length_one_hot`: `14`
- `leader_relative_pos`: `3`
- `next_cards_left`: `14`
- `across_cards_left`: `14`
- `prev_cards_left`: `14`
- `history_next`: `52`
- `history_across`: `52`
- `history_prev`: `52`
- `is_leader`: `1`
- `relative_pass_mask`: `3`
- `is_next_warning`: `1`
- `is_northern_rule`: `1`

合计：

```text
52 + 52 + 10 + 14 + 3 + 14 + 14 + 14 + 52 + 52 + 52 + 1 + 3 + 1 + 1 = 335
```

因此，单个状态张量的形状是：

```text
obs: [335]
```

如果写成批量形式：

```text
obs_batch: [N, 335]
```

其中 `N` 在不同阶段有不同含义：

- 推理时，`N = 当前合法动作数`
- 训练时，`N = T * B`

### obs 的每一部分表示什么

#### 1. `current_hand: 52`

表示当前玩家自己的手牌，多热编码。

- 编码顺序：`[D, C, H, S] x [3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A, 2]`
- 某个位置为 `1` 表示手里有这张牌
- 为 `0` 表示没有这张牌

意义：

- 这是最核心的私有信息
- 决定当前能构造哪些合法动作
- 也决定某些动作打出后会留下什么手型

#### 2. `last_action: 52`

表示上一手实际打出的牌，也是 52 维多热编码。

意义：

- 告诉模型当前这一轮桌面上需要压牌的对象是什么
- 如果全 0，通常意味着当前玩家是新一轮领出者

#### 3. `action_type_one_hot: 10`

表示上一手牌型，取值包括：

- `none`
- `single`
- `pair`
- `triple`
- `straight`
- `flush`
- `full_house`
- `four_of_a_kind`
- `straight_flush`
- `bomb`

意义：

- 告诉模型当前是在接单张、对子、三张，还是 5 张牌型
- 对北方规则尤其重要，因为是否允许炸弹、是否必须同类型压牌，与当前牌型直接相关

#### 4. `action_length_one_hot: 14`

表示上一手出牌张数，索引范围 `0..13`。

意义：

- 与牌型一起确定当前比较关系
- 能显式告诉模型“这一手是空手、4 张、还是 5 张”

#### 5. `leader_relative_pos: 3`

表示最近一次非 `pass` 的领出者相对当前玩家的位置：

- 下家
- 对家
- 上家

如果当前就是自己领出，则这一段全 0。

意义：

- 帮助模型理解当前轮次的攻防关系
- 同样的桌面牌面，如果是上家领出还是下家领出，策略含义可能不同

#### 6. `next_cards_left / across_cards_left / prev_cards_left: 14 x 3`

分别表示：

- 下家剩余牌数
- 对家剩余牌数
- 上家剩余牌数

每个都是 14 维 one-hot，对应 `0..13`。

意义：

- 让模型知道谁快出完了
- 决定是该抢控、压制，还是可以保守处理

#### 7. `history_next / history_across / history_prev: 52 x 3`

分别表示三个相对位置玩家已经打出过哪些牌，用 52 维多热编码表示。

意义：

- 这是公开信息的压缩表示
- 模型可据此推测外面还剩哪些关键牌没有出现
- 对判断对手是否还保留大牌、炸弹、顺子材料等有帮助

#### 8. `is_leader: 1`

标记当前是否轮到自己领出：

- `1` 表示当前无需压牌，是新一轮主动出牌
- `0` 表示当前需要接上一手

意义：

- 领出与接牌是两种完全不同的决策模式
- 领出时更关注出牌结构和控场
- 接牌时更关注是否值得压、是否必须压

#### 9. `relative_pass_mask: 3`

表示在当前这一轮中，下家、对家、上家是否已经 `pass`。

意义：

- 这告诉模型当前轮次里还有哪些人保留响应权
- 如果某些玩家已经过牌，本轮后续博弈空间会缩小

#### 10. `is_next_warning: 1`

表示下家是否只剩一张牌。

意义：

- 这是一个强风险信号
- 需要模型在快要被“跑牌”时更积极压制或抢先出完

#### 11. `is_northern_rule: 1`

表示当前是否使用北方规则。

意义：

- 同一状态在不同规则下合法动作和价值判断可能不同
- 例如炸弹是否能出、是否必须同类型压牌，都受规则影响

### obs 的设计思路

可以把 `obs` 粗略分成四类信息：

- 自己当前可支配资源：`current_hand`
- 当前轮次要解决的问题：`last_action`、`action_type_one_hot`、`action_length_one_hot`、`is_leader`
- 对手公开状态：`cards_left`、`played_history`、`relative_pass_mask`
- 风险和规则提示：`is_next_warning`、`is_northern_rule`

它的目标不是完整还原所有隐藏信息，而是把对当前出牌决策最重要的可见上下文压缩成一个固定长度向量。

## 动作表示：140 维

每个合法动作不会直接用一个稠密动作 id one-hot 表示，而是编码成动作特征向量。编码逻辑在 `ChudadiEnv._action_ids_to_features()`。

当前 140 维动作特征由以下部分组成：

- `action_bits`: `52`
- `next_hand_bits`: `52`
- `action_type_features`: `10`
- `action_main_rank`: `13`
- `action_kicker_rank`: `13`

合计：

```text
52 + 52 + 10 + 13 + 13 = 140
```

单个动作特征张量形状：

```text
action_feature: [140]
```

批量形式：

```text
action_feature_batch: [N, 140]
```

这里的设计非常关键，因为锄大地的理论动作空间是 `2**52`，无法像小动作空间游戏那样直接输出全动作分数表。当前实现转而枚举“当前合法动作”，再逐个打分。

### act 的每一部分表示什么

这里的 `act` 指的是送进模型的动作特征 `action_feature`，而不是原始动作 id。

#### 1. `action_bits: 52`

表示这一步具体打出了哪些牌，使用 52 维多热编码。

意义：

- 明确告诉模型动作本身是什么
- 区分例如“出单张 A”和“出单张 3”这类同牌型但价值完全不同的动作

#### 2. `next_hand_bits: 52`

表示执行该动作之后，当前玩家手里还剩哪些牌。

它的构造方式是：

```text
next_hand_bits = current_hand_bits - action_bits
```

再裁剪到 `0/1`。

意义：

- 这是当前动作特征里最有价值的一部分之一
- 它显式告诉模型“出完这手之后，自己的牌型结构会变成什么”
- 网络不仅看到“现在打了什么”，还能看到“为此付出的手牌结构代价”

例如：

- 打出一个高价值对子
- 可能会破坏后续顺子、葫芦或控场能力
- `next_hand_bits` 能直接暴露这一点

#### 3. `action_type_features: 10`

表示该候选动作本身的牌型，和 `obs` 中上一手牌型使用同一套类别。

意义：

- 让模型不用完全依赖 52 位牌面自己再推导一次牌型
- 更容易学习不同类型动作的通用价值模式

#### 4. `action_main_rank: 13`

表示该动作用于比较大小的主点数。

例如：

- 单张 A，对应主点数 A
- 对子 5，对应主点数 5
- 三张 8，对应主点数 8
- 葫芦 `KKK44`，主点数是 K
- 铁支 `QQQQ3`，主点数是 Q
- 顺子/同花/同花顺，主点数通常是最大牌点数

意义：

- 把“这手牌主要有多大”显式编码出来
- 降低模型从原始牌面中自行抽取大小关系的负担

#### 5. `action_kicker_rank: 13`

表示动作的副点数或次关键点数。

例如：

- 顺子/同花/同花顺：通常用次大牌点数
- 葫芦：对子那部分的点数
- 铁支：单牌脚牌的点数
- 对不适用的牌型，这一段为全 0

意义：

- 进一步补足动作内部结构信息
- 区分一些主点数相同但副结构不同的牌型价值

### act 的设计思路

动作特征不是只描述“这一步出了什么牌”，而是同时描述三件事：

- 当前动作本身是什么：`action_bits`
- 执行后自己还剩什么：`next_hand_bits`
- 这手牌在规则中的结构属性：`action_type_features`、`action_main_rank`、`action_kicker_rank`

因此它比简单的动作 id 更适合泛化。模型可以学到的不是“某个固定 id 好不好”，而是“某种结构的动作在某种局面下值不值得出”。

## DMC 网络结构

`DMCNet` 定义在 `rlcard/agents/dmc_agent/model.py`。

网络结构是一个纯 MLP：

```text
input_dim = state_dim + action_dim = 335 + 140 = 475

475
 -> Linear(475, 512) + ReLU
 -> Linear(512, 512) + ReLU
 -> Linear(512, 512) + ReLU
 -> Linear(512, 512) + ReLU
 -> Linear(512, 512) + ReLU
 -> Linear(512, 1)
 -> flatten
```

默认隐藏层配置：

```python
mlp_layers=[512, 512, 512, 512, 512]
```

前向传播定义是：

```python
def forward(self, obs, actions):
    obs = torch.flatten(obs, 1)
    actions = torch.flatten(actions, 1)
    x = torch.cat((obs, actions), dim=1)
    values = self.fc_layers(x).flatten()
    return values
```

因此张量形状变化为：

```text
obs:      [N, 335]
actions:  [N, 140]
concat:   [N, 475]
fc out:   [N, 1]
flatten:  [N]
```

输出是标量动作价值，而不是动作分布。

## 推理时数据如何流转

推理入口在 `DMCAgent.predict()`。

### 单步推理流程

给定某一时刻当前玩家的状态 `state`：

1. 取出 `state['obs']`
2. 取出 `state['legal_actions']`
3. 将所有合法动作的特征堆叠
4. 将当前状态复制到与合法动作数相同的批量维度
5. 送入网络得到每个合法动作的分数
6. 取最大值对应的动作

### 张量形状追踪

设当前有 `K` 个合法动作。

原始数据：

```text
obs: [335]
legal_actions: {action_id: feature[140]}
```

整理后：

```text
action_keys: [K]
action_values: [K, 140]
obs_repeated: [K, 335]
```

进入网络：

```text
obs: [K, 335]
actions: [K, 140]
concat: [K, 475]
network output: [K]
```

最后：

```text
best_action = argmax(values)
```

### 行为策略

训练期间 `DMCAgent.step()` 使用 epsilon-greedy：

- 以概率 `exp_epsilon` 随机选一个合法动作
- 否则选当前打分最高的合法动作

默认：

```text
exp_epsilon = 0.01
```

评估期间 `eval_step()` 则直接贪心选最大值。

## 对局流程如何产生训练样本

环境完整对局流程在 `Env.run()`。

一局对局期间，环境会记录每个玩家的轨迹：

```text
[state0, action0, state1, action1, ..., final_state]
```

这份轨迹是按玩家分开的：

```text
trajectories[player_id]
```

每局结束后，环境返回：

- `trajectories`
- `payoffs`

其中 `payoffs[player_id]` 是该玩家本局最终收益。

在锄大地中，payoff 的来源是 `ChuDaDiJudger.judge_payoffs()`：

- 胜者得到所有输家罚分之和
- 输家得到自己的负罚分

因此这是一个终局监督信号，而不是逐步 reward。

## actor 进程如何把对局变成监督数据

采样逻辑在 `rlcard/agents/dmc_agent/utils.py::act()`。

actor 会不断重复：

```python
trajectories, payoffs = env.run(is_training=True)
```

然后对每个玩家 `p`：

1. 遍历该玩家的所有决策点
2. 取对应状态中的 `obs`
3. 将该步动作 id 再编码成 `action_feature`
4. 使用最终 `payoffs[p]` 作为该步的监督 target
5. 写入共享内存 buffer

关键代码语义等价于：

```text
for each decision (s, a) in this episode for player p:
    target = final_payoff_of_player_p
```

也就是说，一个玩家在同一局中的所有动作样本都共享同一个最终标签。

这正是当前 DMC 实现的核心特点。

## 训练 buffer 的张量形状

buffer 规格在 `create_buffers()` 中定义。

对每个玩家，每个 buffer slot 保存：

- `done`: `[T]`
- `episode_return`: `[T]`
- `target`: `[T]`
- `state`: `[T, 335]`
- `action`: `[T, 140]`

其中：

- `T = unroll_length`
- 默认 `T = 100`

所以单个 buffer slot 的形状为：

```text
done           [T]
episode_return [T]
target         [T]
state          [T, 335]
action         [T, 140]
```

`get_batch()` 会把 `B` 个 buffer slot 沿着新维度堆叠，得到：

```text
state  [T, B, 335]
action [T, B, 140]
target [T, B]
done   [T, B]
```

其中：

- `B = batch_size`
- 默认 `B = 32`

## learner 侧张量流转

在 `learn()` 中，batch 会先被展平时间维和 batch 维：

```python
state = torch.flatten(batch['state'], 0, 1).float()
action = torch.flatten(batch['action'], 0, 1).float()
target = torch.flatten(batch['target'], 0, 1)
```

因此张量形状变为：

```text
state  [T * B, 335]
action [T * B, 140]
target [T * B]
```

随后送入网络：

```text
concat -> [T * B, 475]
output -> [T * B]
```

默认参数下：

- `T = 100`
- `B = 32`

则 learner 单次更新看到的样本量是：

```text
T * B = 3200
```

对应形状：

```text
state  [3200, 335]
action [3200, 140]
target [3200]
```

## 训练目标是什么

损失函数定义非常直接：

```python
loss = ((pred_values - targets)**2).mean()
```

也就是对每个样本做均方误差回归：

```text
L = mean((Q_theta(s, a) - G)^2)
```

其中：

- `Q_theta(s, a)` 是网络输出
- `G` 是该玩家在整局结束后的最终 payoff

这里的 `pred_values` 指的是网络对每个 `(s, a)` 样本输出的预测价值：

```text
pred_values = Q_theta(s, a)
```

`targets` 则是该玩家在该局最终得到的 payoff。在当前 DMC 实现中，同一局中该玩家的所有决策样本共享同一个最终目标值。

因此：

```text
pred_values - targets
```

表示“当前预测值与真实终局回报之间的误差”。

这不是：

- DQN 的 Bellman target
- actor-critic 的 advantage
- policy gradient 的 log-prob objective

它就是最直接的 Monte Carlo 回报监督。

## 对局结果如何指导权重变化方向

设某个训练样本为：

- 状态 `s`
- 动作 `a`
- 终局回报 `G`

网络当前预测为：

```text
q = Q_theta(s, a)
```

损失为：

```text
L = (q - G)^2
```

于是有两种核心情况。

### 情况 1：网络低估了这个动作

如果：

```text
q < G
```

说明这个 `(s, a)` 的价值被低估。

梯度下降会推动参数朝着“增大 `Q_theta(s, a)`”的方向更新，也就是：

- 让未来再次遇到类似 `(s, a)` 时
- 该动作被打得更高分
- 更容易在 `argmax` 中被选中

### 情况 2：网络高估了这个动作

如果：

```text
q > G
```

说明这个 `(s, a)` 的价值被高估。

梯度下降会推动参数朝着“减小 `Q_theta(s, a)`”的方向更新，也就是：

- 降低未来类似 `(s, a)` 的评分
- 使它更不容易被贪心策略选中

### 直观理解

如果某类动作经常出现在高 payoff 的对局里，这类 `(s, a)` 对应的输出会逐渐被抬高。

如果某类动作经常出现在低 payoff 的对局里，这类 `(s, a)` 对应的输出会逐渐被压低。

因此，终局结果通过监督学习的方式，逐步塑造了动作排序。

## 这种 credit assignment 的特点

当前实现把一个玩家在整局中的所有决策点都赋予同一个最终 payoff，因此 credit assignment 是比较粗粒度的。

优点：

- 实现简单
- 与动态合法动作空间兼容
- 不需要 bootstrap，也不需要 target network

局限：

- 早期动作和末尾关键动作共享同一标签
- 样本噪声较大
- “哪一步真正导致输赢”不会被细粒度地区分出来

也就是说，这个方法能学到“长期上更好的出牌倾向”，但不能像强 credit assignment 方法那样精准定位关键动作。

## 多玩家模型组织方式

`DMCModel` 会为每个玩家位置各创建一个独立 `DMCAgent`。

对于 `chudadi`：

- 4 个玩家位置
- 4 套独立网络参数
- 4 个 optimizer

因此当前结构是：

```text
seat 0 -> model 0
seat 1 -> model 1
seat 2 -> model 2
seat 3 -> model 3
```

这不是共享参数的单模型多位置方案。

这意味着：

- 不同座位可以学出不同策略偏好
- 但参数量和样本需求更大
- 如果未来希望增强泛化，座位共享参数会是一个可考虑方向

## actor 与 learner 如何协同

`DMCTrainer` 使用多进程 actor + 多线程 learner 的结构。

流程如下：

1. 创建 actor 模型副本并放入共享内存
2. actor 进程不断自博弈采样整局数据
3. 采样结果写入共享 buffer
4. learner 线程从 buffer 中取 batch
5. 对对应座位的模型做一次 MSE 更新
6. 更新后的参数再同步回 actor 模型副本

这是一种异步采样、同步参数传播的训练方式。

## 与 Q-learning 的差异

当前实现虽然输出的是“动作值”，但它不是标准 Q-learning：

- 没有 `r + gamma * max_a' Q(s', a')`
- 没有 target network
- 没有 bootstrapped target
- 没有 replay 中的 next-state Bellman backup

所以更准确的说法是：

```text
Monte Carlo supervised regression over action values
```

而不是严格意义上的 DQN。

## 一个完整的张量流示例

假设某一时刻某玩家有 `K = 18` 个合法动作。

### 推理阶段

```text
obs                          [335]
18 个合法动作特征            [18, 140]
复制 obs 后                  [18, 335]
拼接后                       [18, 475]
MLP 输出                     [18]
argmax 选出 1 个动作
```

### 训练阶段

假设：

- `T = 100`
- `B = 32`

则：

```text
batch['state']               [100, 32, 335]
batch['action']              [100, 32, 140]
batch['target']              [100, 32]

flatten 后:
state                        [3200, 335]
action                       [3200, 140]
target                       [3200]

concat                       [3200, 475]
pred_values                  [3200]
MSE loss                     scalar
```

## 训练阶段张量形状逐步追踪

下面按一次完整训练更新来追踪各个量从环境采样到参数更新时的形状变化。

### 1. actor 从环境拿到整局结果

actor 调用：

```python
trajectories, payoffs = env.run(is_training=True)
```

这里拿到的还不是统一张量，而是 Python 结构：

- `trajectories`: 长度为 4 的列表，每个元素是一名玩家的轨迹
- `payoffs`: 长度为 4 的列表，每个元素是该玩家的终局收益标量

对单个玩家 `p`，轨迹结构近似为：

```text
[state0, action0, state1, action1, ..., final_state]
```

其中每个决策样本里：

- `state['obs']`: `[335]`
- `action`: 一个动作 id

### 2. actor 把动作 id 编码成动作特征

在 `act()` 中，会对每个决策点执行：

```python
obs = state['obs']
action = env.get_action_feature(action_id, state)
```

此时单个训练样本可以理解为：

```text
obs            [335]
action_feature [140]
target         []
done           []
episode_return []
```

其中：

- `target = payoffs[p]`
- `done` 表示该样本是否是该玩家本段 episode 片段里的终止位置
- `episode_return` 通常只在终止位置上记录最终 payoff，其余位置多为 0

### 3. 写入单个共享 buffer slot

当某个玩家累计到至少 `T` 个样本时，会写入一个共享 buffer slot。

单个 slot 中字段形状为：

```text
done           [T]
episode_return [T]
target         [T]
state          [T, 335]
action         [T, 140]
```

这里各字段 dtype 约定为：

- `done`: `bool`
- `episode_return`: `float32`
- `target`: `float32`
- `state`: `int8`
- `action`: `int8`

因此，actor 阶段写入 buffer 时，状态和动作特征仍然是离散编码后的整数张量。

### 4. get_batch 堆叠多个 buffer slot

`get_batch()` 会从 full queue 中取出 `B` 个 slot，并在 `dim=1` 上堆叠：

```python
batch = {
    key: torch.stack([buffers[key][m] for m in indices], dim=1)
    for key in buffers
}
```

于是 batch 中各字段变成：

```text
done           [T, B]
episode_return [T, B]
target         [T, B]
state          [T, B, 335]
action         [T, B, 140]
```

默认参数下：

- `T = 100`
- `B = 32`

对应就是：

```text
done           [100, 32]
episode_return [100, 32]
target         [100, 32]
state          [100, 32, 335]
action         [100, 32, 140]
```

### 5. learner 展平时间维和 batch 维

在 `learn()` 中首先执行：

```python
state = torch.flatten(batch['state'], 0, 1).float()
action = torch.flatten(batch['action'], 0, 1).float()
target = torch.flatten(batch['target'], 0, 1)
```

因此形状变化为：

```text
state   [T, B, 335] -> [T*B, 335]
action  [T, B, 140] -> [T*B, 140]
target  [T, B]      -> [T*B]
```

默认参数下：

```text
state   [100, 32, 335] -> [3200, 335]
action  [100, 32, 140] -> [3200, 140]
target  [100, 32]      -> [3200]
```

注意这里：

- `state.float()` 和 `action.float()` 把 `int8` 特征转成浮点输入网络
- `target` 保持为一维回报目标

### 6. 网络内部的形状变化

将 `state` 与 `action` 输入 `agent.forward(state, action)` 后，网络内部的形状变化为：

```text
obs         [T*B, 335]
actions     [T*B, 140]
concat      [T*B, 475]
h1          [T*B, 512]
h2          [T*B, 512]
h3          [T*B, 512]
h4          [T*B, 512]
h5          [T*B, 512]
linear out  [T*B, 1]
flatten     [T*B]
```

最终得到：

```text
pred_values [T*B]
```

默认参数下：

```text
pred_values [3200]
```

### 7. 误差与 loss 的形状

loss 逐元素计算后再求平均：

```text
pred_values           [3200]
targets               [3200]
pred_values - targets [3200]
(...)^2               [3200]
mean(...)             []
```

因此：

- `pred_values - targets` 是每个样本各自的预测误差
- `loss` 是整个 batch 的标量损失

这个标量会进入：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 8. 训练时几个关键量的区别

训练里最容易混淆的几个量可以这样区分：

- `obs`
  - 含义：状态特征
  - 单样本形状：`[335]`
- `action`
  - 含义：动作特征
  - 单样本形状：`[140]`
- `targets`
  - 含义：最终 payoff 监督值
  - 单样本形状：`[]`
- `pred_values`
  - 含义：模型预测的 `Q_theta(s, a)`
  - 单样本形状：`[]`
- `pred_values - targets`
  - 含义：预测误差
  - 单样本形状：`[]`
- `loss`
  - 含义：整个 batch 的平均平方误差
  - 形状：`[]`

### 9. 一次训练更新的形状总览

```text
单个决策样本:
obs            [335]
action_feature [140]
target         []

写入单个 buffer slot:
state          [T, 335]
action         [T, 140]
target         [T]

堆叠成 batch:
state          [T, B, 335]
action         [T, B, 140]
target         [T, B]

flatten 后:
state          [T*B, 335]
action         [T*B, 140]
target         [T*B]

网络输出:
pred_values    [T*B]

误差与损失:
pred_values - targets     [T*B]
(pred_values - targets)^2 [T*B]
loss                      []
```

## 关于 reward shaping 的说明

`rlcard/utils/utils.py::reorganize()` 中存在一个 `_chudadi_low_single_reward()`，会对某些“小单牌且该点数仅有一张”的出法加上 `+0.01`。

但需要注意：

- 当前 DMC 主训练路径并没有调用 `reorganize()`
- DMC 训练主路径是 `act() -> env.run() -> 直接写 target=payoff`

因此，这个 shaping 目前并没有实际进入 `chudadi` 的 DMC 主训练闭环。

如果后续要显式引入中间奖励，需要在 DMC 的采样或 target 构造路径中接入，而不是只在 `reorganize()` 中定义。

## 当前方案的优点与局限

### 优点

- 非常适合巨大且动态的合法动作空间
- 无需输出全动作分数表
- 训练目标简单稳定，工程实现成本低
- 容易导出为 ONNX，线上推理形式也直接

### 局限

- 终局回报回填到所有动作，credit assignment 粗糙
- 四个座位独立建模，样本效率偏低
- 没有利用 bootstrap，学习速度可能慢于更强的值学习方法
- 每次推理都需要对所有合法动作逐个评分，合法动作较多时开销上升

## 总结

当前锄大地模型的本质可以概括为：

- 状态编码成 `335` 维
- 每个合法动作编码成 `140` 维
- 将二者拼接成 `475` 维输入
- 用 5 层 `512` 宽度的 MLP 输出一个标量动作价值
- 推理时对所有合法动作做逐项打分并取最大值
- 训练时用整局最终 payoff 对该玩家本局所有 `(s, a)` 做 Monte Carlo 回归

对模型权重的影响方向也很直接：

- 高 payoff 样本会把对应动作分数往上推
- 低 payoff 样本会把对应动作分数往下压

经过大量自博弈后，网络逐渐学到“在某类局面下，哪些出牌动作更可能导向更好的终局结果”。
