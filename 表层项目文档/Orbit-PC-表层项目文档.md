# Orbit-PC 表层项目文档

> 归属：Orbit 主线表层
> 结构位置：Orbit → 表层 → Orbit-PC

## 项目卡片

| 项目项 | 内容 |
| --- | --- |
| 产品名 | Orbit-PC |
| 表层 | 电脑版本地训练、对话、模型管理和 OpenAI 兼容 API |
| 项目主线 | 是，Orbit-PC 是 Orbit 的电脑版主线 |
| 主线项目文档 | `/Users/tim/Desktop/YUNSH/Orbit/Orbit-项目文档.md` |
| 实际代码 | `/Users/tim/Desktop/YUNSH/Orbit/orbit/` |
| 模型文档 | `/Users/tim/Desktop/YUNSH/Orbit/orbit/Orbit模型文档/` |
| 训练文档 | `/Users/tim/Desktop/YUNSH/Orbit/orbit/Orbit训练方式与参数数据.md` |

## 表层职责

Orbit-PC 负责把 Orbit 的底层架构和训练层提供给普通电脑用户，包括训练/微调、
人工与 AI 语料、参数推荐、样本/Token/步数计算、checkpoint、按需加载、
本地对话和 OpenAI 兼容 API。它不因为 Phone 或 XR 存在就改变自己的模型、
训练记录、版本或发布资产。
