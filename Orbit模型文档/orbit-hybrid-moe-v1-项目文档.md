# orbit-hybrid-moe-v1 项目文档

> 归属：独立 Orbit 项目的底层模型架构
> 产品主线：Orbit-PC

## 1. 项目卡片

| 项目项 | 内容 |
| --- | --- |
| 架构名称 | `orbit-hybrid-moe-v1` |
| 项目归属 | Orbit 底层模型架构，与 OCA 并列 |
| 实现目录 | `/Users/tim/Desktop/YUNSH/Orbit/orbit/orbit/` |
| 模型文档目录 | `/Users/tim/Desktop/YUNSH/Orbit/orbit/Orbit模型文档/` |
| 当前状态 | 自研实验性字节级因果语言模型，已有代码和小规模 checkpoint |
| 主线表层 | Orbit-PC |
| 其他表层 | Orbit-Phone、Orbit-XR 通过各自适配层使用产品能力，不改变底层归属 |

## 2. 结构

```text
UTF-8 字节输入（256 类）
  → Embedding
  → DepthResidual
  → DeltaAttention / GatedMLA
  → RMSNorm
  → LatentMoE（8 routed experts，Top-2，1 shared expert）
  → 共享 Embedding 的语言模型输出头
  → 预测下一个 UTF-8 字节
```

当前 300M 配置约 284,070,912 参数、28 层、12 个头、latent dimension 256、
最大序列长度 2048。1B、3B、7B、14B、38B 是配置模板，不代表已经全部
初始化或预训练。

## 3. 初始化与训练关系

- 预训练第 1 次：没有父 checkpoint 时随机初始化。
- 继续预训练第 2～N 次：加载父 checkpoint，保留父子血缘。
- 微调：必须有兼容父模型，不能没有父模型时静默改成预训练。
- 身份信息必须作为真实训练样本写入，不能由运行时写死回答。
- OCA 是并列的另一底层架构研究项目，不能把 OCA 的实验结果写成此模型能力。

## 4. 关联文档

- [公开模型架构说明](Orbit-模型架构说明.md)
- [模型架构说明](Orbit-模型架构说明.md)
- [训练方式与参数数据](../Orbit训练方式与参数数据.md)
- [OCA 项目文档](OCA-项目文档.md)
