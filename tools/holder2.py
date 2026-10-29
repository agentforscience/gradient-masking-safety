"""GPU memory holder with a control file: keeps `target` GB (1 GB chunks) reserved on the visible GPU.
Jobs lower the target right before they allocate, and raise it back when they finish (src/gpu_hold.py)."""
import os, sys, time, torch
tfile, afile = sys.argv[1], sys.argv[2]
torch.cuda.init()
held = []
while True:
    try:
        target = int(open(tfile).read().strip())
    except Exception:
        target = len(held)
    if target < 0:
        break
    if len(held) > target:          # free one chunk per tick (lock-step handoff)
        held.pop()
        torch.cuda.empty_cache()
        open(afile + ".tmp", "w").write(str(len(held))); os.replace(afile + ".tmp", afile)
        continue
    if len(held) < target:
        try:
            held.append(torch.empty(1 << 30, dtype=torch.uint8, device="cuda"))
            continue
        except Exception:
            torch.cuda.empty_cache()
    open(afile + ".tmp", "w").write(str(len(held))); os.replace(afile + ".tmp", afile)
    time.sleep(0.01 if len(held) == target else 0.2)
