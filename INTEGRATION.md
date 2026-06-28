# Ornith-1.0 — MaoAI / ASI-Genesis 集成层

## 来源
Fork 自 [deepreinforce-ai/Ornith-1](https://github.com/deepreinforce-ai/Ornith-1)（MIT 协议）

## 拆分出的内容
- `server/ornith/adapter.ts` — TypeScript OpenAI-compatible 客户端（MaoAI 用）
- `server/ornith/ornith_coding.py` — Python asyncio 客户端（ASI-Genesis 用）
- `server/ornith/recipes/` — vLLM / SGLang 启动脚本
- `server/ornith/prompts/` — system prompts
- `server/ornith/assets/ornith_logo.png` — Logo

## 三个尺寸

| Size | HF | VRAM | 用途 |
|------|----|------|------|
| 9B   | `deepreinforce-ai/Ornith-1.0-9B`  | 24GB  | 单卡 80GB 推理 |
| 35B  | `deepreinforce-ai/Ornith-1.0-35B` | 80GB  | 多卡全精度 |
| 397B | `deepreinforce-ai/Ornith-1.0-397B`| 800GB | 多卡节点 |

## 快速开始

详见 [`server/ornith/README.md`](server/ornith/README.md)
