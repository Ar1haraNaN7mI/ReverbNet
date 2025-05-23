# ReverberationNet

> 基于角色-乐器网状连接的深度神经网络架构

## 📖 概述

ReverberationNet 是一个创新的深度学习架构，灵感来源于交响乐团的演奏机制。网络采用**网状连接**设计，由19个功能性角色模块和7个乐器层组成，通过门控机制实现动态连接，形成复杂的信息交互网络。

## 🎼 网状架构设计

### 核心组件

**19个角色模块 + 7个融合处理器 + 1个指挥层**

| 序号 | 角色名 | 中文名 | 功能描述 | 连接处理器 |
|------|--------|--------|----------|------------|
| 1 | Eileen | 艾琳 | 高频调节模块 | Harmony |
| 2 | Pluto | 普鲁托 | 大型非线性变换器 | Rhythm |
| 3 | Organ | 管风琴 | 多管道并行处理 | Melody |
| 4 | Harp | 竖琴 | 琶音式序列处理 | Texture |
| 5 | WolfHour | 狼之时刻 | 时域反馈特征建模 | Dynamics |
| 6 | Viola | 中提琴 | 中音域和声分析 | Timbre |
| 7 | Philip | 菲利普 | 底层结构调和器 | Structure |
| 8 | Cello | 大提琴 | 低音域深度共鸣 | Harmony |
| 9 | CircusMaster | 马戏团团长 | 噪声结构解析与正则化 | Rhythm |
| 10 | Bremen | 不莱梅乐队 | 多声道融合组件 | Melody |
| 11 | Zaixian | 在宪 | 附旋律协同控制 | Texture |
| 12 | Elena | 伊莲娜 | 主旋律建模者 | Dynamics |
| 13 | Greta | 格蕾塔 | 节奏结构编码器 | Timbre |
| 14 | Clarinet | 单簧管 | 音色处理 | Structure |
| 15 | Horn | 圆号 | 音域扩展 | Harmony |
| 16 | Tuba | 大号 | 低频增强 | Rhythm |
| 17 | Trombone | 长号 | 滑音处理 | Melody |
| 18 | Violin1 | 第一小提琴 | 主声部 | Texture |
| 19 | Violin2 | 第二小提琴 | 副声部 | Dynamics |

### 融合处理器层

| 处理器名 | 中文名 | 连接角色 | 功能 |
|---------|--------|----------|------|
| Harmony | 和声处理器 | Eileen, Pluto, Organ | 和声特征融合 |
| Rhythm | 节奏处理器 | Harp, WolfHour, Viola | 节奏特征融合 |
| Melody | 旋律处理器 | Philip, Cello, CircusMaster | 旋律特征融合 |
| Texture | 织体处理器 | Bremen, Zaixian, Elena | 织体特征融合 |
| Dynamics | 力度处理器 | Greta, Clarinet, Horn | 力度特征融合 |
| Timbre | 音色处理器 | Tuba, Trombone, Violin1 | 音色特征融合 |
| Structure | 结构处理器 | Violin2, Eileen, Pluto | 结构特征融合 |

## 🔗 网状连接机制

### 三层处理架构

```
输入 → [19个角色模块] → [门控选择] → [7个融合处理器] → [反馈分发] → [角色模块] → Argallia → 输出
      ↑________________  并行处理  ________________↑     ↑___________  网状连接  ___________↑
```

### 连接规则

1. **角色→处理器连接（门控选择）**：
   - 每个角色通过门控机制选择连接到一个融合处理器
   - 门控概率动态调整连接强度
   - 实现自适应的信息路由

2. **处理器→角色连接（反馈机制）**：
   - 每个融合处理器连接到3个角色
   - 简化融合来自多个角色的输入
   - 变分编码确保信息正则化
   - 反馈增强角色的表达能力

3. **全局汇聚**：
   - 所有角色和融合处理器输出汇聚到Argallia指挥层
   - 全局注意力机制提取最终特征
   - 输出标量回归结果

## 🎯 核心特性

### 1. 门控选择机制
```python
# 每个角色模块包含门控层
self.instrument_gate = nn.Linear(d, 1)
gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
```

### 2. 变分编码机制
每个模块都包含变分自编码器：
- **μ (mu)**: 编码均值
- **σ (logvar)**: 编码方差
- **采样**: z = μ + ε × exp(0.5 × logvar)
- **KL散度**: 正则化潜在空间

### 3. 乐器层融合
```python
# 多头注意力融合
fused_output, _ = self.input_fusion(stacked_inputs, stacked_inputs, stacked_inputs)
# 乐器特有处理
processed = self.instrument_processor(fused_output.mean(dim=1, keepdim=True))
# 分发到3个输出
outputs = [distributor(z) for distributor in self.output_distributors]
```

### 4. 自适应维度匹配
网络自动处理不同模块间的维度差异：
```python
if instrument_sum.size(1) != role_output.size(1):
    instrument_sum = F.adaptive_avg_pool1d(
        instrument_sum.transpose(1,2), role_output.size(1)
    ).transpose(1,2)
```

## 🚀 快速开始

### 安装依赖
```bash
pip install torch matplotlib numpy
```

### 基本使用
```python
import torch
from ReverbNet import ReverberationNet

# 创建网状模型
model = ReverberationNet(d=64, num_instruments=7)

# 输入数据 (batch_size, sequence_length, feature_dim)
x = torch.randn(32, 10, 64)

# 前向传播
output, kl_loss, gate_probs = model(x)

print(f"输出形状: {output.shape}")  # (32,) 标量输出
print(f"KL散度: {kl_loss.item()}")
print(f"门控概率数量: {len(gate_probs)}")  # 38个门控（2次角色处理）

# 查看网络连接结构
structure_info = model.get_network_structure()
print(f"总连接数: {structure_info['total_connections']}")
```

### 可视化网状架构
```python
# 生成网状架构图和数据流图
python visualize_architecture.py

# 终端ASCII图
python print_architecture.py
```

## 📊 模型信息

- **总参数量**: ~1,043,413 参数
- **角色层参数**: 733,844 (70.3%)
- **融合处理器参数**: 290,752 (27.9%)
- **指挥层参数**: 18,817 (1.8%)
- **总连接数**: 40个连接
- **网状连接**: 角色↔融合处理器双向连接

## 🎨 架构优势

### 相比顺序连接的改进

| 特性 | 顺序架构 | 网状架构 |
|------|----------|----------|
| **连接方式** | 线性顺序 | 网状交互 |
| **信息流** | 单向传递 | 双向反馈 |
| **表达能力** | 受限于顺序 | 丰富的交互 |
| **自适应性** | 固定连接 | 门控动态选择 |
| **信息保留** | 逐层衰减 | 反馈增强 |

### 网状架构优势

1. **增强非线性能力**: 多层网状连接提供更复杂的特征变换
2. **信息交互丰富**: 角色↔乐器双向连接增强信息流
3. **动态自适应**: 门控机制实现连接的动态调整
4. **反馈增强**: 乐器反馈提升角色表达能力
5. **全局汇聚**: Argallia层实现最优特征整合

## 📁 文件结构

```
ReverbNet-main/
├── ReverbNet.py              # 网状网络定义
├── ExampleCode.py            # 使用示例和训练代码
├── visualize_architecture.py # 网状架构可视化工具
├── print_architecture.py    # 终端架构显示工具
├── README.md                 # 项目说明
├── reverbnet_architecture.png # 网状架构图
├── reverbnet_dataflow.png    # 数据流程图
└── LICENSE                   # 许可证
```

## 🔧 技术细节

### 网状连接映射

```python
# 角色到乐器的映射（循环分配）
role_to_instrument = {
    'eileen': 'piano', 'pluto': 'guitar', 'organ': 'flute',
    'harp': 'trumpet', 'wolfhour': 'drum', ...
}

# 乐器到角色的映射（每个乐器连接3个角色）
instrument_to_roles = {
    'piano': ['eileen', 'pluto', 'organ'],
    'guitar': ['harp', 'wolfhour', 'viola'], ...
}
```

## 📐 数学公式

### 角色模块数学表达式

#### 基础角色模块 (RoleModule)

对于每个角色模块 R_i，处理流程如下：

**1. 角色特有处理**
```
h_i = RoleProcessor_i(x)
```

**2. 融合处理器输入融合**
```
h_i' = h_i + Σ[j∈Processors_i] AdaptivePool(P_j)
```

其中 AdaptivePool 确保维度匹配：
```
AdaptivePool(P_j) = Pool1D(P_j^T, L_target)^T
```

**3. 变分编码**
```
μ_i = W_μ^(i) * h_i'
log(σ_i²) = W_logσ^(i) * h_i'
z_i = μ_i + ε ⊙ exp(0.5 * log(σ_i²)), where ε ~ N(0, I)
```

**4. 门控机制**
```
g_i = sigmoid(W_g^(i) · Mean(h_i', dim=1))
```

**5. 残差连接与层归一化**
```
R_i(x) = LayerNorm(x + z_i)
```

#### 具体角色模块

**Eileen (艾琳) - 高频调节与卷积注意力专家**
```
Conv_Eileen(x) = Conv1D(x^T, kernel=3)^T
h_conv = GELU(Conv_Eileen(x))
h_attn = MultiheadAttention(h_conv, h_conv, h_conv)
Eileen(x) = W_ffn * h_attn
```

**Pluto (普鲁托) - 深度非线性变换专家**
```
h_1 = GELU(W_1^Pluto * x)
h_1' = Dropout(h_1, p=0.1)
h_2 = GELU(W_2^Pluto * h_1')
Pluto(x) = W_3^Pluto * h_2

其中: W_1 ∈ R^(d×4d), W_2 ∈ R^(4d×2d), W_3 ∈ R^(2d×d)
```

**Organ (管风琴) - 多管道并行处理专家**
```
Pipe_j(x) = GELU(W_j^pipe * x), j = 1,2,3,4
h_concat = Concat[Pipe_1(x), Pipe_2(x), Pipe_3(x), Pipe_4(x)]
Organ(x) = W_fusion * h_concat
```

**Harp (竖琴) - LSTM序列建模专家**
```
h_t, c_t = LSTM(x_t, h_{t-1}, c_{t-1})
Harp(x) = [h_1, h_2, ..., h_T]
```

**WolfHour (狼之时刻) - 双向GRU时域专家**
```
h_t_forward = GRU_forward(x_t, h_{t-1}_forward)
h_t_backward = GRU_backward(x_t, h_{t+1}_backward)
WolfHour(x) = h_t_forward + h_t_backward
```

**Viola (中提琴) - 中频谐波分析专家**
```
h_1 = tanh(W_1^Viola * x)
h_2 = W_2^Viola * h_1
Viola(x) = LayerNorm(h_2)
```

**Philip (菲利普) - 结构调和与正则化专家**
```
h_1 = ReLU(W_1^Philip * x), W_1 ∈ R^(d×d/2)
h_2 = W_2^Philip * h_1, W_2 ∈ R^(d/2×d)
h_3 = Dropout(h_2, p=0.2)
Philip(x) = W_3^Philip * h_3
```

**Cello (大提琴) - 低频共鸣与深度特征专家**
```
h_1 = LeakyReLU(W_1^Cello * x, α=0.2)
h_2 = W_2^Cello * h_1
Cello(x) = BatchNorm1D(h_2)
```

**CircusMaster (马戏团长) - 噪声控制与注意力调节专家**
```
NoiseGate(x) = sigmoid(W_noise * x) ⊙ x
h_attn = MultiheadAttention(NoiseGate(x), NoiseGate(x), NoiseGate(x))
CircusMaster(x) = W_control * h_attn
```

**Bremen (不莱梅乐队) - 多声道融合专家**
```
Channel_j(x) = W_j^channel * x, j = 1,2,3
h_multi = Concat[Channel_1(x), Channel_2(x), Channel_3(x)]
h_1 = GELU(W_1^Bremen * h_multi)
Bremen(x) = W_2^Bremen * h_1
```

**Zaixian (在宪) - 附旋律协同控制专家**
```
h_1 = sigmoid(W_1^Zaixian * x)
h_2 = GELU(W_2^Zaixian * h_1)
Zaixian(x) = W_3^Zaixian * h_2
```

**Elena (伊莲娜) - 主旋律建模专家**
```
h_1 = GELU(W_1^Elena * x), W_1 ∈ R^(d×3d)
h_2 = GELU(W_2^Elena * h_1), W_2 ∈ R^(3d×2d)
h_3 = W_3^Elena * h_2, W_3 ∈ R^(2d×d)
Elena(x) = LayerNorm(h_3)
```

**Greta (格蕾塔) - 节奏结构编码专家**
```
h_rhythm = GELU(Conv1D(x^T, kernel=5)^T)
h_tempo = W_tempo * h_rhythm
BeatGate(x) = sigmoid(W_beat * x)
Greta(x) = h_tempo ⊙ BeatGate(x)
```

**Clarinet (单簧管) - 音色处理与频域变换专家**
```
h_1 = Softplus(W_1^Clarinet * x)
h_2 = ELU(W_2^Clarinet * h_1), W_2 ∈ R^(d×d/2)
Clarinet(x) = W_3^Clarinet * h_2, W_3 ∈ R^(d/2×d)
```

**Horn (圆号) - 音域扩展与动态范围专家**
```
h_1 = SiLU(W_1^Horn * x), W_1 ∈ R^(d×4d)
h_2 = SiLU(W_2^Horn * h_1), W_2 ∈ R^(4d×2d)
Horn(x) = W_3^Horn * h_2, W_3 ∈ R^(2d×d)
```

**Tuba (大号) - 低频增强与重低音专家**
```
h_1 = LeakyReLU(W_1^Tuba * x, α=0.3)
h_2 = LeakyReLU(W_2^Tuba * h_1, α=0.3)
h_3 = W_3^Tuba * h_2
Tuba(x) = LayerNorm(h_3)
```

**Trombone (长号) - 滑音处理与连续变换专家**
```
h_slide = GELU(Conv1D(x^T, kernel=7)^T)
h_glide = W_glide * h_slide
SmoothWeight(x) = sigmoid(W_smooth * x)
Trombone(x) = h_glide ⊙ SmoothWeight(x)
```

**Violin1 (第一小提琴) - 主声部领奏专家**
```
h_leader = MultiheadAttention(x, x, x, heads=8)
h_vibrato = GELU(W_vibrato * h_leader)
ExpressionGate(x) = sigmoid(W_expr * x)
Violin1(x) = h_vibrato ⊙ ExpressionGate(x)
```

**Violin2 (第二小提琴) - 副声部和声专家**
```
h_1 = GELU(W_1^Violin2 * x), W_1 ∈ R^(d×2d)
h_2 = W_2^Violin2 * h_1, W_2 ∈ R^(2d×d)
h_3 = Dropout(h_2, p=0.1)
h_4 = W_3^Violin2 * h_3
Violin2(x) = LayerNorm(h_4)
```

### 融合处理器数学表达式（简化版）

#### FusionProcessor 处理流程

对于每个融合处理器 P_k，接收来自连接角色的输入 {R_1, R_2, R_3}：

**1. 简化输入融合**
```
X_avg = (1/3) * Σ[i=1 to 3] R_i
X_fused = W_fusion^P * X_avg
```

**2. 融合处理器特有处理（简化）**
```
h_mean = Mean(X_fused, dim=1)
h_1 = GELU(W_1^P * h_mean), W_1^P ∈ R^(d×2d)
h_proc = W_2^P * h_1, W_2^P ∈ R^(2d×d)
```

**3. 变分编码**
```
μ_P = W_μ^P * h_proc
log(σ_P²) = W_logσ^P * h_proc
z_P = μ_P + ε ⊙ exp(0.5 * log(σ_P²))
```

**4. 输出分发**
```
O_j^P = W_j^dist * z_P, j = 1,2,3
Output_j^P = Expand(O_j^P, L_target)
```

### Argallia指挥层数学表达式

**1. 输入汇总与归一化**
设所有角色输出为 {R_1', R_2', ..., R_19'}，所有处理器输出为 {P_1, P_2, ..., P_7}：

```
AllOutputs = {R_1', R_2', ..., R_19', P_1, P_2, ..., P_7}
```

**2. 维度归一化**
```
O_i_normalized = {
    O_i,                                    if L_i = L_target
    AdaptivePool1D(O_i^T, L_target)^T,     otherwise
}
```

**3. 全局注意力**
```
X_global = Stack([O_1_norm, O_2_norm, ..., O_26_norm], dim=1)
X_attended = MultiheadAttention(X_global, X_global, X_global)
```

**4. 特征提取与最终输出**
```
f_global = Mean(X_attended, dim=1) ∈ R^(B×d)
h_final = GELU(LayerNorm(W_1^Argallia * f_global))
output = W_2^Argallia * h_final ∈ R^B

其中: W_1^Argallia ∈ R^(d×d/2), W_2^Argallia ∈ R^(d/2×1)
```

### 损失函数与优化

#### 总损失函数
```
L_total = L_MSE + α * L_KL
```

#### 均方误差损失
```
L_MSE = (1/B) * Σ[i=1 to B] (y_i - ŷ_i)²
```

#### KL散度正则化
```
L_KL = Σ[i=1 to 19] KL(μ_i, σ_i²) + Σ[j=1 to 7] KL(μ_j^P, σ_j^P²)

其中:
KL(μ, σ²) = -0.5 * Σ[k=1 to d] (1 + log(σ_k²) - μ_k² - σ_k²)
```

#### 门控概率
每个角色的门控概率：
```
p_i^gate = sigmoid(W_g^(i) · (1/L) * Σ[t=1 to L] h_{i,t}')
```

### 网状前向传播完整流程

#### 第一层：角色并行处理
```
R_i^(1) = RoleModule_i(x), i = 1, 2, ..., 19
```

#### 第二层：融合处理器处理
```
P_j = FusionProcessor_j({R_k^(1) : k ∈ Connected(j)}), j = 1, 2, ..., 7
```

#### 第三层：角色反馈处理
```
R_i^(2) = RoleModule_i(R_i^(1), {P_j^(i) : j ∈ Feedback(i)})
```

#### 第四层：全局汇聚
```
y = Argallia({R_1^(2), ..., R_19^(2)}, {P_1, ..., P_7})
```

### 参数复杂度分析

#### 角色模块参数
- **基础角色模块**: 3d² + 3d 参数
- **Eileen**: d² + 16d² + 3d² = 20d² 参数  
- **Pluto**: 4d² + 2d² + d² = 7d² 参数
- **Organ**: 4d² + 4d² = 8d² 参数
- **Harp**: 4d² + d 参数 (LSTM)
- **WolfHour**: 6d² + 2d 参数 (双向GRU)

#### 融合处理器参数
每个融合处理器: d² + 6d² + 3d² = 10d² 参数

#### Argallia层参数
64d² + d²/2 + d/2 + 1 参数

#### 总参数估算
对于 d = 64：
```
Total ≈ 19 × 3d² + 7 × 10d² + 64d² ≈ 1,043,413 参数
```

### 损失函数

```
总损失 = MSE损失 + α × KL散度
```
- **MSE损失**: 主要学习目标
- **KL散度**: 变分正则化项（权重1e-4）

### 网状前向传播流程

1. **第一层**: 19个角色并行处理输入
2. **融合处理器层**: 7个融合处理器融合对应角色输出
3. **第三层**: 角色接收融合处理器反馈再次处理
4. **汇聚层**: Argallia全局注意力汇总

## 🎯 应用场景

- **序列回归**: 时间序列预测、信号处理
- **复杂特征学习**: 多模态信息融合
- **网状信息处理**: 需要丰富信息交互的任务
- **自适应建模**: 需要动态连接调整的场景

## 🔬 实验结果

网状架构展现出优异的性能：
- **参数效率**: 104万参数实现复杂网状连接
- **收敛性**: 良好的损失下降趋势
- **门控分布**: 平均门控概率0.504，分布合理
- **信息保留**: 网状连接有效减少信息损失

## 📈 训练效果预览

### 训练配置
```
模型总参数数量: 1,043,413
训练轮数: 50 epochs
批次大小: 32
学习率: 1e-3
数据量: 1000样本
输入维度: (batch_size, 10, 64)
```

### 损失收敛曲线
```
Epoch   1/50 | Total Loss: 18.115900 | MSE Loss: 0.102766 | KL Loss: 18013.13 | Gate Probs: 38
Epoch   5/50 | Total Loss: 0.359443  | MSE Loss: 0.079483 | KL Loss: 279.96   | Gate Probs: 38
Epoch  10/50 | Total Loss: 0.161604  | MSE Loss: 0.077901 | KL Loss: 83.70    | Gate Probs: 38
Epoch  20/50 | Total Loss: 0.104052  | MSE Loss: 0.075423 | KL Loss: 28.63    | Gate Probs: 38
Epoch  30/50 | Total Loss: 0.082469  | MSE Loss: 0.066810 | KL Loss: 15.66    | Gate Probs: 38
Epoch  40/50 | Total Loss: 0.063780  | MSE Loss: 0.053348 | KL Loss: 10.43    | Gate Probs: 38
Epoch  50/50 | Total Loss: 0.055662  | MSE Loss: 0.047204 | KL Loss: 8.46     | Gate Probs: 38
```

### 性能指标
- **最终训练损失**: 0.055662
- **最终MSE损失**: 0.047204  
- **最终KL散度**: 8.46
- **评估MSE损失**: 0.041924
- **收敛速度**: 快速收敛，前10个epoch损失下降91%

### 门控概率分布
```
各模块门控概率分析:
 1. Eileen       : 0.5203    11. Zaixian     : 0.4777
 2. Pluto        : 0.5134    12. Elena       : 0.2985
 3. Organ        : 0.4653    13. Greta       : 0.4857
 4. Harp         : 0.4987    14. Clarinet    : 0.4911
 5. WolfHour     : 0.4718    15. Horn        : 0.4687
 6. Viola        : 0.4567    16. Tuba        : 0.5163
 7. Philip       : 0.4833    17. Trombone    : 0.5199
 8. Cello        : 0.4883    18. Violin1     : 0.4916
 9. CircusMaster : 0.5306    19. Violin2     : 0.4949
10. Bremen       : 0.5210

平均门控概率: ~0.487 (分布均匀，表明网络充分利用了所有角色模块)
```

### 关键观察
1. **损失快速下降**: 总损失从18.12快速降至0.056，收敛效果良好
2. **MSE稳定**: 回归损失从0.103降至0.047，模型学习效果显著
3. **KL正则化**: KL散度从18013降至8.46，变分编码正常工作
4. **门控均衡**: 所有角色模块的门控概率分布合理(0.30-0.53)，无偏向性
5. **泛化能力**: 评估损失(0.042)低于训练损失，表明良好的泛化性能

## 🤝 贡献

欢迎提交 Issues 和 Pull Requests！

## 📜 许可证

本项目采用 **Mozilla Public License 2.0 (MPL-2.0)** 许可证 - 详见 [LICENSE](LICENSE) 文件。

### ⚠️ 重要使用限制

- **非商业用途**: 本项目仅供学术研究和非商业用途
- **禁止未授权公开**: 未经作者明确许可，**禁止**将本项目用于公开发布或商业用途
- **修改要求**: 如对本项目进行修改，必须在相同许可证下开源修改部分
- **署名要求**: 使用本项目时必须保留原始版权声明和许可证声明

如需商业使用或公开发布，请联系作者获取明确授权。

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和研究者。

---

> 🎵 "19个角色如同音乐家，7个融合处理器如同协调组，Argallia如同指挥家，通过网状连接共同演奏出复杂而美妙的机器学习交响曲。"
