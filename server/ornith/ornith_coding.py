#!/usr/bin/env python3
"""
Ornith-1.0 Agent for ASI-Genesis
--------------------------------
把 Ornith 注册为 ASI-Genesis 的 coding specialist 节点。
后端走 OpenAI 兼容端点 (vLLM / SGLang / Ollama)。

用法:
    from asi_genesis.agents.ornith_coding import OrnithCodingAgent
    agent = OrnithCodingAgent(size="9B")
    result = await agent.run("修复这个 bug: ...")
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import AsyncIterator

import httpx


@dataclass
class OrnithResult:
    answer: str
    reasoning: str
    elapsed_s: float
    usage: dict
    tool_calls: list


class OrnithCodingAgent:
    """ASI-Genesis 适配器: 9B / 35B / 397B"""

    PRESETS = {
        "9B":  {"hf": "deepreinforce-ai/Ornith-1.0-9B",  "ctx": 262144, "vram_gb": 24},
        "35B": {"hf": "deepreinforce-ai/Ornith-1.0-35B", "ctx": 262144, "vram_gb": 80},
        "397B":{"hf": "deepreinforce-ai/Ornith-1.0-397B","ctx": 262144, "vram_gb": 800},
    }

    def __init__(
        self,
        size: str = "9B",
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.6,
        top_p: float = 0.95,
    ):
        if size not in self.PRESETS:
            raise ValueError(f"size must be one of {list(self.PRESETS)}")
        self.size = size
        self.base_url = base_url or os.getenv("ORNITH_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("ORNITH_API_KEY", "EMPTY")
        self.model_name = model_name or os.getenv("ORNITH_MODEL_NAME", "Ornith-1.0")
        self.temperature = temperature
        self.top_p = top_p
        self.preset = self.PRESETS[size]

    async def health(self) -> dict:
        """GET /v1/models"""
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            return {"ok": r.status_code == 200, "models": [m["id"] for m in r.json().get("data", [])]}

    async def run(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> OrnithResult:
        """非流式调用"""
        t0 = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=600) as c:
            r = await c.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": max_tokens,
                },
            )
            r.raise_for_status()
            data = r.json()

        msg = data["choices"][0]["message"]
        return OrnithResult(
            answer=msg.get("content", ""),
            reasoning=msg.get("reasoning_content", ""),
            elapsed_s=time.time() - t0,
            usage=data.get("usage", {}),
            tool_calls=msg.get("tool_calls") or [],
        )

    async def stream(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict]:
        """流式调用，yield reasoning / content / done 事件"""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=600) as c:
            async with c.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": max_tokens,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield {"type": "done", "text": ""}
                        return
                    import json
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("reasoning_content"):
                        yield {"type": "reasoning", "text": delta["reasoning_content"]}
                    if delta.get("content"):
                        yield {"type": "content", "text": delta["content"]}
                    if chunk.get("usage"):
                        yield {"type": "done", "text": "", "usage": chunk["usage"]}


# 便捷工厂
def agent_9b(**kw):  return OrnithCodingAgent(size="9B",  **kw)
def agent_35b(**kw): return OrnithCodingAgent(size="35B", **kw)
def agent_397b(**kw):return OrnithCodingAgent(size="397B",**kw)


# CLI 测试入口
if __name__ == "__main__":
    import asyncio
    import sys

    async def _main():
        agent = OrnithCodingAgent(size=os.getenv("ORNITH_SIZE", "9B"))
        h = await agent.health()
        print(f"[health] {h}")
        if not h["ok"]:
            print("Server unreachable. Start with: ORNITH_SIZE=9B ./recipes/vllm.sh")
            return
        prompt = sys.argv[1] if len(sys.argv) > 1 else "用 Python 写一个 is_prime 函数"
        result = await agent.run(prompt)
        print(f"\n[reasoning] ({len(result.reasoning)} chars)\n{result.reasoning[:500]}")
        print(f"\n[answer]\n{result.answer}")
        print(f"\n[elapsed] {result.elapsed_s:.2f}s  usage: {result.usage}")

    asyncio.run(_main())
