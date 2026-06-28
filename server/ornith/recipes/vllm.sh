#!/usr/bin/env bash
# 启动 Ornith-1.0 vLLM 服务
# 用法: ORNITH_SIZE=9B ./vllm.sh
set -e

ORNITH_SIZE=${ORNITH_SIZE:-9B}
PORT=${PORT:-8000}
TP=${TP:-1}   # tensor parallel size

case "$ORNITH_SIZE" in
  9B)  MODEL="deepreinforce-ai/Ornith-1.0-9B"  ;;
  35B) MODEL="deepreinforce-ai/Ornith-1.0-35B" ;;
  397B) MODEL="deepreinforce-ai/Ornith-1.0-397B" ;;
  *)
    echo "Unknown size: $ORNITH_SIZE (use 9B / 35B / 397B)"
    exit 1
    ;;
esac

echo "Starting Ornith-1.0-$ORNITH_SIZE on port $PORT, TP=$TP"
vllm serve "$MODEL" \
  --served-model-name "Ornith-1.0" \
  --tensor-parallel-size "$TP" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --trust-remote-code
