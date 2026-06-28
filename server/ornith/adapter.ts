/**
 * Ornith-1.0 Adapter
 * ------------------
 * 通用 OpenAI-compatible 客户端，对接 Ornith-1.0 系列 (9B/35B/397B)
 * 服务于 MaoAI 推理层和 ASI-Genesis Agent 层。
 *
 * 特性:
 *  - reasoning_content 字段自动透传 (Ornith 是 reasoning model)
 *  - tool_call 解析 (qwen3_xml / qwen3_coder)
 *  - 流式 + 非流式双模式
 *  - 256K (262,144) 上下文窗口支持
 *
 * 仓库: https://github.com/deepreinforce-ai/Ornith-1
 * HuggingFace: https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B
 *
 * License: MIT (上游) / 本适配器适配 MaoAI 商业用途
 */

import OpenAI from "openai";

// ──────────────────────────────────────────────────────────────
// 类型定义
// ──────────────────────────────────────────────────────────────

export type OrnithSize = "9B" | "35B" | "397B";

export interface OrnithConfig {
  /** 基础 URL，默认 vLLM/SGLang OpenAI 兼容端点 */
  baseUrl?: string;
  /** API key，本地服务用 "EMPTY" 即可 */
  apiKey?: string;
  /** 模型 size，决定默认 checkpoint 和上下文窗口 */
  size?: OrnithSize;
  /** 自定义 model name (覆盖默认) */
  modelName?: string;
  /** 上下文窗口，默认 262144 (256K) */
  maxContext?: number;
  /** 默认温度 (推荐 0.6) */
  temperature?: number;
  /** top_p (推荐 0.95) */
  topP?: number;
  /** top_k (推荐 20) */
  topK?: number;
  /** 是否启用 reasoning parser (强烈建议) */
  reasoningParser?: boolean;
  /** 是否启用 tool-call parser */
  toolCallParser?: "qwen3_xml" | "qwen3_coder" | null;
  /** 请求超时 (ms) */
  timeout?: number;
}

export interface OrnithMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  name?: string;
  tool_call_id?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
}

export interface OrnithResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: "assistant";
      /** 思考链 (reasoning_content) */
      reasoning_content?: string;
      /** 最终回答 */
      content: string;
      tool_calls?: Array<{
        id: string;
        type: "function";
        function: { name: string; arguments: string };
      }>;
    };
    finish_reason: string;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  /** 适配器扩展字段 */
  asi_meta?: {
    size: OrnithSize;
    elapsed_s: number;
    reasoning_chars: number;
  };
}

// ──────────────────────────────────────────────────────────────
// 预设配置 (覆盖 HuggingFace 上的官方 checkpoint)
// ──────────────────────────────────────────────────────────────

export const ORNITH_PRESETS: Record<OrnithSize, { hf: string; gguf: string; ctx: number; vram_gb: number }> = {
  "9B": {
    hf: "deepreinforce-ai/Ornith-1.0-9B",
    gguf: "hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF",
    ctx: 262144,
    vram_gb: 24, // bf16 dense 9B + KV cache
  },
  "35B": {
    hf: "deepreinforce-ai/Ornith-1.0-35B",
    gguf: "hf.co/deepreinforce-ai/Ornith-1.0-35B-GGUF",
    ctx: 262144,
    vram_gb: 80, // MoE 35B bf16
  },
  "397B": {
    hf: "deepreinforce-ai/Ornith-1.0-397B",
    gguf: "hf.co/deepreinforce-ai/Ornith-1.0-397B-GGUF",
    ctx: 262144,
    vram_gb: 800, // MoE 397B bf16, 需要多卡节点
  },
};

// ──────────────────────────────────────────────────────────────
// 适配器类
// ──────────────────────────────────────────────────────────────

export class OrnithAdapter {
  private client: OpenAI;
  private config: Required<Omit<OrnithConfig, "modelName" | "toolCallParser">> & Pick<OrnithConfig, "modelName" | "toolCallParser">;

  constructor(config: OrnithConfig = {}) {
    const size = config.size ?? "9B";
    const preset = ORNITH_PRESETS[size];

    this.config = {
      baseUrl: config.baseUrl ?? process.env.ORNITH_BASE_URL ?? "http://localhost:8000/v1",
      apiKey: config.apiKey ?? process.env.ORNITH_API_KEY ?? "EMPTY",
      size,
      modelName: config.modelName ?? process.env.ORNITH_MODEL_NAME ?? "Ornith-1.0",
      maxContext: config.maxContext ?? preset.ctx,
      temperature: config.temperature ?? 0.6,
      topP: config.topP ?? 0.95,
      topK: config.topK ?? 20,
      reasoningParser: config.reasoningParser ?? true,
      toolCallParser: config.toolCallParser ?? null,
      timeout: config.timeout ?? 300_000,
    };

    this.client = new OpenAI({
      baseURL: this.config.baseUrl,
      apiKey: this.config.apiKey,
      timeout: this.config.timeout,
    });
  }

  /** 健康检查 — /v1/models */
  async health(): Promise<{ ok: boolean; models: string[] }> {
    try {
      const r = await this.client.models.list();
      return { ok: true, models: r.data.map(m => m.id) };
    } catch (e) {
      return { ok: false, models: [] };
    }
  }

  /**
   * 非流式 chat completion
   * 自动提取 reasoning_content → asi_meta
   */
  async chat(
    messages: OrnithMessage[],
    options: { temperature?: number; maxTokens?: number; tools?: unknown[] } = {},
  ): Promise<OrnithResponse> {
    const t0 = Date.now();
    const r: any = await this.client.chat.completions.create({
      model: this.config.modelName!,
      messages: messages as any,
      temperature: options.temperature ?? this.config.temperature,
      top_p: this.config.topP,
      max_tokens: options.maxTokens ?? 4096,
      tools: options.tools as any,
    });

    const msg = r.choices[0]?.message ?? {};
    return {
      ...r,
      asi_meta: {
        size: this.config.size,
        elapsed_s: (Date.now() - t0) / 1000,
        reasoning_chars: (msg.reasoning_content ?? "").length,
      },
    };
  }

  /**
   * 流式 chat completion
   * 同时 yield reasoning_content 和 content
   */
  async *chatStream(
    messages: OrnithMessage[],
    options: { temperature?: number; maxTokens?: number } = {},
  ): AsyncGenerator<{ type: "reasoning" | "content" | "done"; text: string; usage?: any }> {
    const stream: any = await this.client.chat.completions.create({
      model: this.config.modelName!,
      messages: messages as any,
      temperature: options.temperature ?? this.config.temperature,
      top_p: this.config.topP,
      max_tokens: options.maxTokens ?? 4096,
      stream: true,
      stream_options: { include_usage: true },
    });

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta ?? {};
      if (delta.reasoning_content) {
        yield { type: "reasoning", text: delta.reasoning_content };
      }
      if (delta.content) {
        yield { type: "content", text: delta.content };
      }
      if (chunk.usage) {
        yield { type: "done", text: "", usage: chunk.usage };
      }
    }
    yield { type: "done", text: "" };
  }

  /**
   * 适配 MaoAI 现有 aiStream.ts 的非流式 JSON 协议
   * (MaoAI 后端目前把 ASI-Genesis 的非流式结果切成 SSE)
   */
  async chatAsAsiGenesis(messages: OrnithMessage[]): Promise<{ id: string; content: string; reasoning?: string; elapsed: number }> {
    const r = await this.chat(messages);
    const m = r.choices[0]?.message;
    return {
      id: r.id,
      content: m?.content ?? "",
      reasoning: m?.reasoning_content,
      elapsed: r.asi_meta?.elapsed_s ?? 0,
    };
  }
}

// ──────────────────────────────────────────────────────────────
// 工厂函数 (按 size)
// ──────────────────────────────────────────────────────────────

export const ornith9b = (overrides?: Partial<OrnithConfig>) => new OrnithAdapter({ ...overrides, size: "9B" });
export const ornith35b = (overrides?: Partial<OrnithConfig>) => new OrnithAdapter({ ...overrides, size: "35B" });
export const ornith397b = (overrides?: Partial<OrnithConfig>) => new OrnithAdapter({ ...overrides, size: "397B" });

export default OrnithAdapter;
