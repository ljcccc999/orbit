# Orbit 自研模型架构说明

> 归属：独立 Orbit 项目。本文档描述当前 Orbit 本地训练模型，不适用于 YUNSH OS 内置 Orbit、Orbit App 移动客户端或 OCA-Research。

## 1. 架构身份

- 产品名称：Orbit
- 当前架构名称：`orbit-hybrid-moe-v1`
- 架构性质：Orbit 自研实验架构
- 当前状态：已实现模型代码和 300M 训练 checkpoint；尚未完成大规模基础预训练
- Kimi/K3 关系：不是 Kimi K3 的严格复现，不加载 Kimi/K3 权重，也不能使用“K3mini”作为当前架构名称

MLA、MoE、RMSNorm 等术语是公开的通用模型组件名称。当前代码中的自定义组合不能仅凭名称证明来自某一份技术报告。若未来要声明“参考 K3 公开架构思路后独立实现”，必须先增加逐项报告映射、实现差异、配置/权重验证和基准测试记录。

## 2. 当前数据流

```text
UTF-8 文本
  → 字节编码（0–255，共 256 类）
  → Embedding
  → Orbit Backbone 多层模块
  → 共享 Embedding 权重的 LM Head
  → 预测下一个 UTF-8 字节
```

当前训练器是 tokenizer-free 的 UTF-8 byte-level causal language model：一个 UTF-8 字节作为一个内部训练 token。它不是 BPE 或 SentencePiece token，不能把该计数直接当作外部模型的 token 计数。

## 3. Backbone 层结构

每一层大致按以下顺序工作：

```text
当前隐藏状态
  → DepthResidual 深度残差融合
  → RMSNorm
  → DeltaAttention 或 GatedMLA
  → 残差相加
  → RMSNorm
  → LatentMoE
  → 残差相加
```

### 3.1 DepthResidual

`DepthResidual` 保存并堆叠部分历史深度来源，通过可学习 query 计算 softmax 权重，将历史层级表示与当前表示融合。它不是 Transformer 的普通短连接，而是当前 Orbit 自定义的深度残差模块。

### 3.2 DeltaAttention

`DeltaAttention` 包含：

- Q/K/V 投影；
- 深度可分离短卷积，用于局部序列信息；
- Q/K 归一化；
- 门控输出投影；
- 默认使用 PyTorch fused causal scaled dot-product attention。

代码还保留一个带 beta、decay 和 recurrent state 的逐 token Delta 路径，用于兼容/调试；当前 `fast_attention=true` 时不会走该慢路径。

### 3.3 GatedMLA

`GatedMLA` 是 Orbit 自定义的潜变量压缩注意力模块：

- Q 直接从隐藏状态投影；
- 先把隐藏状态压缩到 latent dimension；
- 再从 latent 表示上投影 K/V；
- 使用 causal attention；
- 最后通过 sigmoid gate 和输出投影融合。

它是 Orbit 的自定义实现，名称中的 MLA 不代表已经复现 Kimi K3 的 MLA 细节。

### 3.4 LatentMoE

每个 `LatentMoE` 层包含：

- 一个路由器；
- 8 个 routed experts；
- 每个 token 激活 Top-2 experts；
- 一个 latent bottleneck；
- 1 个 shared expert；
- SiTUGLU 门控前馈网络；
- MoE 路由负载均衡损失。

当前实现只计算被 Top-2 选中的专家，避免把稀疏 MoE 实际执行成所有专家都计算的 dense MoE。

## 4. 输出和训练目标

- 输出头与输入 Embedding 权重共享；
- 目标是预测下一个 UTF-8 字节；
- 主损失为 causal language modeling cross-entropy；
- 额外加入少量 MoE router balance loss；
- 训练身份样本属于训练语料，不允许推理代码根据“你是谁”写死回答。

当前 `OrbitForCausalLM` 没有启用视觉编码器。配置中的 `vision_width`、`vision_layers` 等字段是预留配置，不代表当前模型已经具备图像输入能力。

## 5. 模型规模档位

App 中的规模档位是配置模板：

| 档位 | 层数 | 说明 |
|---|---:|---|
| 300M | 28 | 当前已有本地训练 checkpoint，约 2.84 亿参数 |
| 1B | 28 | 配置模板，未随 App 预初始化权重 |
| 3B | 32 | 配置模板，未随 App 预初始化权重 |
| 7B | 36 | 配置模板，未随 App 预初始化权重 |
| 14B | 42 | 配置模板，未随 App 预初始化权重 |
| 38B | 48 | 配置模板，未随 App 预初始化权重 |

这些档位不会在 App 启动时全部分配内存。只有用户选择档位并开始训练或加载兼容 checkpoint 时，程序才会创建或加载对应权重；内存不足时必须在分配前阻止任务并建议远程 GPU。

## 6. 初始化和训练血缘

### 预训练第 1 次

没有父 checkpoint 时，训练入口按选定档位创建 `OrbitForCausalLM(cfg)`，由 PyTorch 对参数进行随机初始化，然后从用户语料开始训练。

### 继续预训练第 2～N 次

选择上一轮 Orbit checkpoint，加载父模型权重，保留训练血缘并使用新的本轮训练配置。模型规模必须与父模型一致。

### 微调

必须选择已有 Orbit checkpoint 或经过真实兼容性验证的外部基础模型。微调不能把模型规模偷偷改成另一个 M/B 档位，也不能在没有父模型时静默降级为从零预训练。

## 7. 与 OCA-Research 的边界

`/Users/tim/Desktop/YUNSH/Orbit/orbit/OCA-Research/` 是独立的研究实验目录，拥有自己的 `orbit/model.py`、配置、论文和 checkpoint。OCA 是自研、尚未实现完成的世界模型研究架构；它不是当前 `orbit/` 公开仓库中的桌面 Orbit 对话模型实现，也没有证明已经理解真实物理世界。两者不能混用架构名称、权重、训练记录或测试结论。

OCA 的研究目标可以表述为“目标成为行业首个理解物理世界的架构”，但这
只是待验证的研究/产品目标，不是已经确认的行业事实。使用更强的公开表述
前，需要先完成先前技术检索、独立物理世界基准、公平对比和第三方验证。

## 8. 当前源码入口

- [模型实现](../orbit/model.py)
- [规模配置](../orbit/config.py)
- [训练入口](../orbit/train.py)
- [Orbit训练方式与参数数据](../Orbit训练方式与参数数据.md)
- [旧版研究依据文档](../docs/训练数据质量与参数建议-研究依据.md)
