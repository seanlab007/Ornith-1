# Ornith 部署脚本集合

## vLLM (推荐)

详见 [vllm.sh](vllm.sh) — 支持 9B / 35B / 397B 三档

## SGLang

详见 [sglang.sh](sglang.sh)

## Ollama (本地 9B)

```bash
ollama run hf.co/deepreinforce-ai/Ornith-1.0-9B-GGUF
```

## llama.cpp

```bash
llama-server -hf deepreinforce-ai/Ornith-1.0-9B-GGUF --port 8000 -c 262144
```
