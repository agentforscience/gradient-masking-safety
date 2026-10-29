"""Generate greedy responses of one target model to every prompt in datasets/prompts.jsonl.
Memory: takes `need` GB from the GPU holder (tools/holder2.py) under a file lock, keeps it as a placeholder tensor,
and frees the placeholder only at the instant vLLM measures free memory (in-process engine) -> no startup race.
Usage: python generate.py MODEL_KEY [LIMIT]"""
import os, sys, json, fcntl, time, atexit
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HOME", "/data/transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from common import *
from models import MODELS

key = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
cfg = MODELS[key]
LOGS = ROOT / "logs"
out_path = RES / "gens" / (f"{key}.jsonl" if LIMIT is None else f"{key}.test.jsonl")
os.makedirs(out_path.parent, exist_ok=True)
lockf = open(str(out_path) + ".lock", "w")
try:
    fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    print("LOCKED", out_path); sys.exit(0)

prompts = read_jsonl(DATA / "prompts.jsonl")
if LIMIT:
    import random; random.Random(0).shuffle(prompts); prompts = prompts[:LIMIT]
done = {r["pid"] for r in read_jsonl(out_path)} if out_path.exists() else set()
todo = [p for p in prompts if p["pid"] not in done]
print(f"{key}: {len(done)} done, {len(todo)} to do", flush=True)
if not todo:
    sys.exit(0)

import torch
torch.cuda.init()
TF, AF = LOGS / "hold_g1.target", LOGS / "hold_g1.actual"
need = cfg["need"]
PLACE = []

def _grab(timeout):
    t = time.time()
    while time.time() - t < timeout:
        try:
            PLACE.append(torch.empty(1 << 30, dtype=torch.uint8, device="cuda")); return True
        except Exception:
            pass
    return False

with open(LOGS / "handoff.lock", "w") as hl:
    fcntl.flock(hl, fcntl.LOCK_EX)
    t0 = time.time()
    while int(open(TF).read()) < need:      # wait until enough is parked in the holder
        fcntl.flock(hl, fcntl.LOCK_UN); time.sleep(5); fcntl.flock(hl, fcntl.LOCK_EX)
    # lock-step handoff: holder frees one 1 GB chunk, we grab it immediately (first one also hosts our CUDA context)
    while len(PLACE) < need and time.time() - t0 < 900:
        cur = int(open(TF).read()); open(TF, "w").write(str(cur - 1))
        while int(open(AF).read() or 0) > cur - 1:
            time.sleep(0.001)
        if not _grab(3):
            print("lost a chunk to another process", flush=True)
    if len(PLACE) > 0:
        PLACE.pop(); torch.cuda.empty_cache()   # leave room for the CUDA context / non-torch overhead
    print(f"acquired {len(PLACE)}/{need} GB in {time.time()-t0:.1f}s", flush=True)

def give_back():
    with open(LOGS / "handoff.lock", "w") as hl:
        fcntl.flock(hl, fcntl.LOCK_EX)
        cur = int(open(TF).read()); open(TF, "w").write(str(cur + need))
atexit.register(give_back)

import vllm.v1.worker.gpu_worker as gw
_Base = gw.MemorySnapshot
class _Snap(_Base):
    def measure(self):
        if PLACE:
            PLACE.clear(); torch.cuda.empty_cache()
        return super().measure()
gw.MemorySnapshot = _Snap

from vllm import LLM, SamplingParams
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
llm = LLM(model=cfg["hf"], dtype="auto", max_model_len=2048, gpu_memory_utilization=(need - 0.4) / total_gb,
          kv_cache_memory_bytes=int(cfg["kv"] * 2**30), enforce_eager=True, max_num_seqs=128, seed=SEED,
          trust_remote_code=False, enable_prefix_caching=True)
sp = SamplingParams(temperature=0.0, max_tokens=256, seed=SEED)
ctk = {"enable_thinking": False} if cfg.get("thinking_off") else None

def run(convs):
    outs = llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs=ctk)
    return [o.outputs[0].text for o in outs]

single = [p for p in todo if len(p["turns"]) == 1]
multi = [p for p in todo if len(p["turns"]) > 1]
CH = 800
t0 = time.time()
with open(out_path, "a") as f:
    for i in range(0, len(single), CH):
        ch = single[i:i + CH]
        texts = run([[{"role": "user", "content": p["turns"][0]}] for p in ch])
        for p, t in zip(ch, texts):
            f.write(json.dumps(dict(pid=p["pid"], model=key, response=t, history=[]), ensure_ascii=False) + "\n")
        f.flush(); print(f"{i+len(ch)}/{len(single)} single  {time.time()-t0:.0f}s", flush=True)
    if multi:
        convs = [[] for _ in multi]; hist = [[] for _ in multi]
        for r in range(max(len(p["turns"]) for p in multi)):
            for c, p in zip(convs, multi):
                c.append({"role": "user", "content": p["turns"][r]})
            texts = run(convs)
            for c, h, t in zip(convs, hist, texts):
                c.append({"role": "assistant", "content": t}); h.append(t)
        for p, h in zip(multi, hist):
            f.write(json.dumps(dict(pid=p["pid"], model=key, response=h[-1], history=h[:-1]), ensure_ascii=False) + "\n")
        print(f"multi-turn done {time.time()-t0:.0f}s", flush=True)
print("DONE", key, flush=True)
