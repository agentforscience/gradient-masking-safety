#!/bin/bash
# Stop worker 1 once its model queue is done; worker 2 (TAKEOVER_PID) inherits its GPU memory.
cd "$(dirname "$0")/.."
until grep -qE "(FINISHED|FAILED) qwen2.5-32b$" logs/worker_g1.log; do sleep 10; done
touch logs/STOP_worker
