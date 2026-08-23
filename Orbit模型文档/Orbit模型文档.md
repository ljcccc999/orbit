# Orbit模型文档

这是电脑版 Orbit 的模型文档目录首页。该目录只描述独立 Orbit 的模型、
训练方式、初始化语义、OCA 研究边界和验证结果，不适用于 Orbit-XR 或
Orbit-Phone。

## 当前模型

- `orbit-hybrid-moe-v1`：当前桌面 Orbit 的自研实验性字节级语言模型。
- 300M、1B、3B、7B、14B、38B：配置档位；不是全部已经初始化或预训练的模型。
- 训练第 1 次没有父模型时随机初始化；继续训练和微调加载父 checkpoint。

## OCA 状态

OCA（Orbit Continuum Architecture）是 Orbit 的自研世界模型研究架构，
当前**未实现完成**，没有证明已经理解真实物理世界。项目目标是探索并争取
成为行业首个理解物理世界的架构；“行业首个”目前只是目标/待验证主张，
不是已确认事实。OCA 不属于当前桌面 Orbit 对话模型，不能混用 checkpoint。

OCA 研究目录：

`/Users/tim/Desktop/YUNSH/Orbit/orbit/OCA-Research/`

## 文档索引

- [Orbit 自研模型架构技术报告](Orbit-架构技术报告.md)
- [Orbit 模型架构说明](Orbit-模型架构说明.md)
- [OCA 研究 README](../OCA-Research/README.md)

每次模型实现、训练方式、OCA 研究状态或验证结论发生变化，都要同步更新
本目录、电脑版 Orbit 项目文档和 Codex 的 Orbit 长期记忆；不能把未验证的
实验结果写成已实现能力。
