# Orbit 三层架构项目文档

> Orbit 的统一结构：**架构 → 训练 → 表层**
> 电脑版主线正式名称：**Orbit-PC**

## 1. 总览

```text
Orbit
├─ 架构层
│  ├─ OCA
│  └─ orbit-hybrid-moe-v1
├─ 训练层
│  └─ 训练方式、参数、样本、Token、数据与验证
└─ 表层
   ├─ Orbit-PC（电脑版，Orbit 主线）
   ├─ Orbit-Phone（iPhone/HarmonyOS）
   └─ Orbit-XR（YUNSH OS 空间/系统集成）
```

这三层属于同一个 Orbit 产品体系，但每层有不同职责和文档边界。不能把
表层功能、训练结果或某个平台的硬件测试，反向写成底层架构已经具备的能力。

## 2. 架构层

架构层包含两个并列的底层项目，各自有独立项目文档：

| 底层项目 | 状态 | 文档 |
| --- | --- | --- |
| OCA | 自研世界模型研究架构，尚未实现完成 | [OCA-项目文档](Orbit模型文档/OCA-项目文档.md) |
| `orbit-hybrid-moe-v1` | 自研实验性字节级因果语言模型 | [orbit-hybrid-moe-v1-项目文档](Orbit模型文档/orbit-hybrid-moe-v1-项目文档.md) |

OCA 的“目标成为行业首个理解物理世界的架构”是研究/产品目标，不是已验证
事实。OCA 不能与 `orbit-hybrid-moe-v1` 混用 checkpoint、参数或测试结论。

## 3. 训练层

训练层负责把用户授权的数据和选定底层架构连接起来，包括：

- 预训练（从零开始）；
- 继续预训练（第 2～N 次）；
- 微调和父模型血缘；
- 人工文献与 AI 教师样本；
- 样本、字符、UTF-8 字节、训练 Token、Batch、梯度累计和优化器步数；
- 300M～38B 规模规划、参数推荐、内存、时间、ETA、验证集和训练质量边界。

统一文档为：[Orbit训练方式与参数数据](Orbit训练方式与参数数据.md)。

## 4. 表层

表层是用户实际使用的产品入口，共有三个：

| 表层 | 定位 | 文档 |
| --- | --- | --- |
| Orbit-PC | 电脑版，Orbit 主线；训练、对话、模型和本地 API | [Orbit-PC 表层文档](表层项目文档/Orbit-PC-表层项目文档.md) |
| Orbit-Phone | iPhone/HarmonyOS 移动端表层 | [Orbit-Phone 表层文档](表层项目文档/Orbit-Phone-表层项目文档.md) |
| Orbit-XR | YUNSH OS 空间/系统集成表层 | [Orbit-XR 表层文档](表层项目文档/Orbit-XR-表层项目文档.md) |

Orbit-PC 是 Orbit 主线，拥有独立的主线项目文档：
`/Users/tim/Desktop/YUNSH/Orbit/Orbit-项目文档.md`。

Orbit-Phone 和 Orbit-XR 在 YUNSH 中共用一个合并项目文档：
`/Users/tim/Desktop/YUNSH/项目文档/Orbit-Phone-XR-项目文档.md`。

## 5. 文件归属

- 本目录及其子目录属于 Orbit 公开项目的文档入口；
- YUNSH 中的 Phone/XR 合并文档负责移动端与系统集成端的生态边界；
- YUNSH OS 实际源码仍以 `/Users/tim/Desktop/YUNSH/YUNSH OS/yunsh-os/` 为唯一来源；
- 训练、模型和表层文档不能替代源码、checkpoint 或真实硬件验证。
