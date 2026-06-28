# Ornith-1.0 — Agentic Coding LLM 系列集成

> **Fork**: [seanlab007/Ornith-1](https://github.com/seanlab007/Ornith-1) ⇐ [deepreinforce-ai/Ornith-1](https://github.com/deepreinforce-ai/Ornith-1)
> **License**: MIT (上游) · 本适配器：MaoAI 内部使用
> **HF Collection**: https://huggingface.co/collections/deepreinforce-ai/ornith-10

## 是什么

Ornith-1.0 是一组**自改进 (self-improving)** 的 Agentic Coding 模型，基于 Gemma 4 / Qwen 3.5 后训练，**专为代码生成 + 工具调用设计**。在 Terminal-Bench 2.1、SWE-Bench、NL2Repo、OpenClaw 基准上**同尺寸开源 SOTA**。

| 尺寸 | 架构 | 上下文 | 显存 (bf16) | 适用 |
|------|------|--------|-------------|------|
| **9B**  | Dense  | 256K | 24GB   | 单卡 80GB 推理 / 微调 |
| **35B** | MoE    | 256K | 80GB   | 多卡全精度服务 |
| **397B**| MoE    | 256K | 800GB  | 多卡节点服务 |

每个尺寸都提供：bf16、FP8、GGUF 三种格式。

## 拆分原则

本目录 (`server/ornith/`) 是从 upstream 拆出**集成层代码**，用于：
- **MaoAI** 后端 — 把 Ornith 接入到统一推理网关，作为 cloud-coding 模型
- **ASI-Genesis** — 把它注册为本地 Agent 的 coding specialist 节点

**不**包含：
- 模型权重（请从 HF 下载）
- 训练代码（upstream 没开放）
- 评测代码（upstream 没说）

## 集成点

| 项目 | 入口 | 触发方式 |
|------|------|---------|
| **MaoAI** | `server/ornith/adapter.ts` → `server/models.ts::ornith-1.0` 系列 | `useAsiGenesis: true` 路由 |
| **ASI-Genesis** | `asi-genesis/agents/ornith_coding.py` | Node `asi-genesis-ornith` |
| **WorkBuddy** | 通过 ASI-Genesis 间接调用 | 选 ASI-Genesis → 选 coding 任务 |

## 启动服务（vLLM）

```bash
MODEL=deepreinforce-ai/Ornith-1.0-9B  # 或 35B / 397B
vllm serve $MODEL \
  --served-model-name Ornith-1.0 \
  --tensor-parallel-size 1 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code
```

## 启动服务（Ollama，本地 9B）

```bash
ollama run hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF
# OpenAI 兼容端点自动开在 :11434/v1
```

## MaoAI 端调用

```typescript
import { ornith9b } from "./ornith/adapter";

const ornith = ornith9b({ baseUrl: "http://localhost:8000/v1" });
const r = await ornith.chat([
  { role: "user", content: "用 TypeScript 写一个 debounce 函数" },
]);
console.log(r.choices[0].message.content);
```

## ASI-Genesis 端调用

```python
from asi_genesis.adapters.ornith import OrnithCodingAgent

agent = OrnithCodingAgent(size="9B")
result = await agent.run("Fix this TypeScript bug: ...")
print(result.answer)
print(result.reasoning)  # 思考链
```

## 上游 Benchmarks（节选）

| 模型 | SWE-Bench Verified | Terminal-Bench 2.1 | NL2Repo |
|------|---------------------|--------------------|---------|
| Ornith-1.0-9B    | **69.4** | 43.1 | 27.2 |
| Ornith-1.0-35B   | **75.6** | 64.2 | 34.6 |
| Ornith-1.0-397B  | **82.4** | 77.5 | 48.2 |
| Qwen3.5-397B (基线) | 76.4   | 53.5 | 36.8 |
| Claude Opus 4.7  | 80.8 | 70.3 | - |

## 引用

```bibtex
@misc{ornith-1.0,
  title = {{Ornith-1.0}: Agentic Coding, Open to All},
  url = {https://deep-reinforce.com/ornith_1_0.html},
  author = {{DeepReinforce Team}},
  year = {2026}
}
```
