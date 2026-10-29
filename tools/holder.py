"""Reserve GPU memory in 1 GB chunks until RELEASE file appears (shared-GPU startup-race guard)."""
import os, sys, time, torch
gpu, gb, flag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
torch.cuda.set_device(0)
held = []
while len(held) < gb and not os.path.exists(flag):
    try:
        held.append(torch.empty(1 << 30, dtype=torch.uint8, device="cuda"))
    except Exception:
        time.sleep(0.5)
print("holding", len(held), "GB", flush=True)
while not os.path.exists(flag):
    time.sleep(0.2)
