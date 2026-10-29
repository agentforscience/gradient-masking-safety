#!/bin/bash
# HTTP judge (Qwen3-8B server): judge each model once its generation is complete (all prompts, or a .done marker).
cd "$(dirname "$0")"
N=$(wc -l < ../datasets/prompts.jsonl)
for m in qwen2.5-0.5b qwen3-0.6b llama3.2-1b qwen2.5-1.5b qwen2.5-3b llama3.2-3b phi3-mini mistral-7b gemma2-9b qwen2.5-7b llama3.1-8b qwen2.5-14b llama2-7b-chat qwen2.5-32b; do
  until [ -f ../results/gens/$m.done ] || { [ -f ../results/gens/$m.jsonl ] && [ "$(wc -l < ../results/gens/$m.jsonl)" -ge "$N" ]; }; do sleep 20; done
  python3 judge.py $m
done
echo ALLJUDGED
