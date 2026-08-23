# Orbit 自研模型架构技术报告

> 版本：0.1（研究与实现边界说明）  
> 归属：独立 Orbit 桌面项目  
> 状态：当前 Orbit 主模型已实现实验性代码；OCA 仍是未实现完成的世界模型研究架构

## 摘要

Orbit 是面向本地训练、推理和开放 API 的独立 AI 工作室。当前桌面 Orbit
使用自研的 `orbit-hybrid-moe-v1` 作为实验性字节级因果语言模型。它不是
Kimi K3 的严格复现，也没有加载 Kimi/K3 权重；本文档不把通用的 MLA、MoE
或 RMSNorm 术语当作来源证明。

Orbit 另有一条名为 **OCA（Orbit Continuum Architecture）** 的研究路线。
OCA 计划把持久世界状态、对象槽位、动作条件状态转移和未来想象作为独立
结构加入模型，用于研究理解物理世界的可能路径。OCA 当前仍未实现为可用
世界模型，现有原型和合成世界测试不能证明它已经理解真实物理世界。
“自研、目标成为行业首个理解物理世界的架构”是 Orbit 的研究/产品目标，
不是已经完成或经独立基准验证的事实。

## 1. 产品与边界

### 1.1 三个平台分开

| 平台 | 目录 | 边界 |
| --- | --- | --- |
| Orbit | `/Users/tim/Desktop/YUNSH/Orbit/` | 电脑版本地训练、推理、桌面 App 和 OpenAI 兼容 API |
| Orbit-XR | `/Users/tim/Desktop/YUNSH/Orbit-XR/` | YUNSH OS 内置的空间/系统集成版本 |
| Orbit-Phone | `/Users/tim/Desktop/YUNSH/Orbit-Phone/` | iPhone/HarmonyOS 移动端 |

三者不能互相冒充模型、权重、版本、测试结果或发布资产。本文只描述
独立 Orbit 的模型研究；Orbit-XR 的系统源码仍以
`/Users/tim/Desktop/YUNSH/YUNSH OS/yunsh-os/` 为唯一来源，Orbit-Phone
的移动端源码和真机测试也不归入本文。

### 1.2 两个模型研究分支

- `orbit-hybrid-moe-v1`：当前桌面 Orbit 的实验性语言模型，已经有模型代码
  和小规模训练 checkpoint，但不等于完成了大规模基础预训练。
- OCA：独立的世界模型研究分支，目录为
  `/Users/tim/Desktop/YUNSH/Orbit/orbit/OCA-Research/`。它目前不被桌面
  Orbit 对话运行时加载，也不能与 `orbit-hybrid-moe-v1` 的 checkpoint、
  参数和测试结论混用。

## 2. Orbit 主模型：`orbit-hybrid-moe-v1`

### 2.1 输入与训练目标

当前输入层采用 tokenizer-free 的 UTF-8 byte-level 表示：每个 UTF-8 字节
映射到 0–255 的 256 类词表。模型执行因果语言建模，预测下一个字节；
因此 Orbit 内部的“token”计数是字节数，不能直接当成 BPE 或 SentencePiece
模型的 token 计数。

```text
UTF-8 文本
  → 0–255 字节编码（256 类）
  → Embedding
  → Orbit Backbone
  → 共享 Embedding 权重的 LM Head
  → 下一个 UTF-8 字节
```

训练目标包括 causal language-model cross-entropy，以及用于稀疏专家路由
的负载均衡项。Orbit 身份样本可以进入训练语料，但运行时不得检测“你是谁”
并写死答案；导出模型是否能回答身份问题，必须由 checkpoint 的真实权重
决定。

### 2.2 主干结构

每一层的概念顺序为：

```text
隐藏状态
  → DepthResidual 深度残差融合
  → RMSNorm
  → DeltaAttention 或 GatedMLA
  → 残差相加
  → RMSNorm
  → LatentMoE
  → 残差相加
```

#### DepthResidual

DepthResidual 保存部分历史深度来源，通过可学习 query 计算融合权重，将
历史层级表示与当前表示组合。它是 Orbit 自定义的深度残差模块，不代表
标准 Transformer 中的普通短连接。

#### 混合注意力

- `DeltaAttention` 包含 Q/K/V 投影、局部深度可分离卷积、Q/K 归一化和
  门控输出；默认使用 PyTorch fused causal attention。
- `GatedMLA` 先把隐藏状态压缩到 latent 表示，再从 latent 表示投影 K/V，
  通过 causal attention 和 sigmoid gate 输出。
- 旧的逐 token recurrent Delta 路径只用于兼容/调试，不是默认快速训练路径。

这里的“MLA”是当前 Orbit 的自定义实现名，不能据此声称复现任何外部模型
的 MLA 细节。

#### LatentMoE

- 8 个 routed experts；
- 每个 token 激活 Top-2 experts；
- 叠加 1 个 shared expert；
- 专家内部使用 Orbit 的 SiTUGLU 门控前馈网络；
- 训练时加入路由负载均衡损失；
- 只计算选中的专家，避免把稀疏 MoE 实际执行成所有专家的 dense 计算。

输入 Embedding 与输出 LM Head 共享权重，以降低参数量并保持字节级输出
结构一致。

### 2.3 当前规模配置

当前 300M 档实际配置约为 284,070,912 参数、28 层、12 个注意力头、latent
dimension 256、最大序列长度 2048。App 中的 1B、3B、7B、14B、38B 是
架构配置模板，不是 App 启动时已经初始化或预训练好的六个模型；只有用户
选择档位并开始训练或加载 checkpoint 时才分配/读取权重。

规模档位是工程配置，不等于模型能力。随机初始化的 300M 模型经过少量
步骤只能证明训练链路，不能宣称已经具备正常聊天、编程或世界理解能力。

## 3. 初始化、预训练与微调

### 3.1 预训练第 1 次

没有父 checkpoint 时，按用户选择的 Orbit 配置创建模型并随机初始化权重，
再从用户授权的文献、教材、代码、科学/数学资料和少量任务/身份样本训练。
真正的大规模预训练需要与参数量相匹配的数据规模和算力，少量教师样本
不能替代基础语料。

### 3.2 继续预训练第 2～N 次

选择上一轮 Orbit checkpoint 后，加载父模型权重，在新增且经过清洗、去重、
验证集隔离的语料上继续训练。模型规模必须保持兼容，训练历史必须保留
父子血缘，不能把继续训练伪装成全新随机初始化。

### 3.3 微调

微调必须选择已有的兼容父模型。已训练过的 Orbit checkpoint 可以作为父模型；
外部模型只有在 tokenizer、架构、词表、权重格式和推理/训练适配全部通过
真实验证后才能进入微调列表。没有父模型时，界面和后端都必须阻止微调，
不能静默降级成预训练。

## 4. OCA：未实现的世界模型研究架构

### 4.1 研究目标

OCA 的目标不是把更多文本塞进普通语言模型，而是显式建模“观察—状态—
干预—未来”的关系：

```text
观察 O_t
  → 感知 P
  → 对象槽位绑定 B
  → 持久连续状态 S_t
  → 动作条件状态转移 T
  → 多步未来想象 S_(t+1...t+n)
  → 语言/动作解码
```

核心假设可以写成：

```text
S_t = U(S_(t-1), B(P(O_t)))
未来状态 = T(当前状态, 动作)
```

候选训练目标包括语言损失、对象属性状态损失、干预后的动力学预测、
未受影响对象保持损失、循环一致性和不确定性校准。

### 4.2 当前实现边界

OCA 目前包含研究原型代码、合成世界数据、tiny/小规模训练入口和 7B
结构配置草案。它可以用于检查代码是否能运行、状态是否能传递以及损失是否
能计算，但当前尚未完成以下关键目标：

- 没有证明能理解真实世界的物体、时间、因果关系或物理规律；
- 没有通过与参数匹配 Transformer 的隐藏物体追踪对比；
- 没有证明对未见过的对象/动作组合泛化；
- 没有证明多步想象能提升规划；
- 没有完成充分的消融实验、长期状态稳定性测试和真实物理基准测试；
- OCA 研究 checkpoint 不是当前桌面 Orbit 的聊天模型。

因此 OCA 在项目状态中必须标注为：**未实现完成 / 研究中 / 不可宣称已
理解物理世界**。

### 4.3 “行业首个”表述边界

Orbit 可以把 OCA 描述为“自研、目标成为行业首个理解物理世界的架构”，
但不能写成“已经是行业首个”或“已经实现理解物理世界”。要使用后者，
至少需要公开可复现的任务定义、与现有方法的公平对比、独立评测、先前技术
检索和同行/第三方验证。目前这些证据尚未完成。

### 4.4 OCA 研究目录

OCA 已从 YUNSH 顶层 Orbit 资料中归入电脑版 Orbit 项目目录：

`/Users/tim/Desktop/YUNSH/Orbit/orbit/OCA-Research/`

它是 Orbit 公开项目目录中的独立研究子目录，但仍与主模型代码、主模型
checkpoint、训练历史和生产对话运行时分开。

## 5. 验证与复现要求

每次涉及模型架构或 OCA 的变更，都要分别记录：

1. 源码/静态检查；
2. 小规模单元测试和 smoke test；
3. 训练 loss、验证集指标和数据来源；
4. checkpoint、配置和训练血缘；
5. 独立对话或世界状态评测；
6. 尚未验证的范围。

“能生成 checkpoint”“loss 下降”“代码测试通过”和“模型能正常聊天/理解
物理世界”是不同结论，不能互相替代。

## 6. 研究路线

1. 先保持 `orbit-hybrid-moe-v1` 与 OCA 的代码、权重、文档和运行时边界；
2. 为 OCA 固定可复现的合成世界基准和参数匹配基线；
3. 完成状态、对象槽位、干预和多步想象的单项消融；
4. 通过基准后再评估更大规模和真实/模拟物理数据；
5. 只有在独立验证和先前技术检索完成后，才更新“行业首个”等公开表述。

## 7. 相关文件

- Orbit 模型结构说明：`Orbit-模型架构说明.md`
- OCA 研究代码：`../OCA-Research/`
- Orbit 项目文档：`../../Orbit-项目文档.md`
- Orbit-XR 边界文档：`/Users/tim/Desktop/YUNSH/Orbit-XR/Orbit-XR-项目文档.md`
- Orbit-Phone 边界文档：`/Users/tim/Desktop/YUNSH/Orbit-Phone/Orbit-Phone-项目文档.md`

