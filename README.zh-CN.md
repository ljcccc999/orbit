# Orbit

**Orbit** 是 YUNSH 的本地 AI 训练平台：在自己的电脑上训练、微调并运行自己的模型，并通过本机对话、OpenAI 兼容本机 API 和 **Orbit Code** 编码 Agent 使用它们。

## 三层体系

- **架构层** — 两条自研研究线：
  - **orbit-hybrid-moe-v1**：Orbit 的实验性混合 MoE 架构（字节级词表、GatedMLA、带残差深度的潜变量 MoE）。
  - **OCA（Orbit Continuum Architecture）**：面向具有持久空间状态的世界模型架构的研究计划。OCA 是研究目标，不是已交付的产品功能。
- **训练层** — 预训练、继续预训练、微调、数据、checkpoint 与质量边界。
- **表层** — 你实际使用的产品：**Orbit-PC**（桌面主线）、**Orbit-Phone**、**Orbit-XR**（内置于 YUNSH OS）。

## 它能做什么

- 从零训练模型，或微调已有的 Orbit checkpoint
- 在本机生成训练数据，支持或不支持 AI 辅助
- 本机推理与 OpenAI 兼容本机 API（`/v1/chat/completions`、`/v1/responses`）
- 通过审核制贡献包与可选 Orbit Hub 协作
- Orbit 对话与 Orbit Code 共用一套可排序的模型库：API 与本地模型共享默认项，训练模式保留专用的本地模型操作
- 每次重新打开自动进入新对话并创建 Orbit 工作区；选择完全访问后仍可操作工作区之外的内容
- 每次 Orbit Code 回答完成后显示文件变更卡片，可审核差异或经确认后执行带冲突检查的安全撤销

## 产品表层

- **Orbit-PC** — 桌面训练、对话、Orbit Code Agent 与本机 API
- **Orbit-Phone** — Orbit 手机伴侣
- **Orbit-XR** — 集成在 YUNSH OS 中的空间桌面

## 源码开放说明

Orbit 的模型实现与详细技术文档属于 YUNSH 内部资产，不在本仓库公开；本仓库仅提供产品概述。
