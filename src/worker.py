"""Single long-lived GPU worker: takes over the memory parked in tools/holder2.py (lock-step, 1 GB chunks), keeps it as
placeholder tensors, and runs the target models one after another IN-PROCESS. For each model the placeholder is freed only
at the instant vLLM measures free memory, and re-grabbed right after the engine is torn down -> no startup races with other
tenants of the shared GPU.  Usage: python worker.py QUEUE_FILE   (one model key per line; re-read after every model)"""
import os, sys, json, time, gc, fcntl
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HOME", "/data/transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from common import *
from models import MODELS
import torch

LOGS = ROOT / "logs"
TF, AF = LOGS / "hold_g1.target", LOGS / "hold_g1.actual"
PLACE = []
def _ctx():
    while True:                           # CUDA context first (needs ~0.5 GB free; retry while the GPU is full)
        try:
            torch.zeros(1, device="cuda"); return
        except Exception:
            time.sleep(0.2)
_ctx()

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def grab(timeout=2.0):
    t = time.time()
    while time.time() - t < timeout:
        try:
            PLACE.append(torch.empty(1 << 30, dtype=torch.uint8, device="cuda")); return True
        except Exception:
            time.sleep(0.001)
    return False

if os.environ.get("TAKEOVER_PID"):
    # take over the memory of a previous worker: wait for it to exit, then grab 1 GB chunks in a tight loop
    pid, want = int(os.environ["TAKEOVER_PID"]), int(os.environ.get("TAKEOVER_GB", "34"))
    while os.path.exists(f"/proc/{pid}"):
        time.sleep(0.002)
    t = time.time()
    while len(PLACE) < want and time.time() - t < 20:
        grab(0.05)
    if (LOGS / "STOP_worker").exists():
        os.remove(LOGS / "STOP_worker")       # that file was meant for the previous worker
else:
    # take over everything parked in the holder (tools/holder2.py), chunk by chunk
    while True:
        cur = int(open(TF).read())
        if cur <= 0:
            break
        open(TF, "w").write(str(cur - 1))
        t = time.time()
        while int(open(AF).read() or 0) > cur - 1 and time.time() - t < 10:
            time.sleep(0.001)
        grab()
    open(TF, "w").write("-1")                 # holder exits
log(f"worker holds {len(PLACE)} GB")
TOTAL = len(PLACE)

# Memory hand-off to vLLM WITHOUT ever returning memory to the driver: placeholders are dropped into PyTorch's caching
# allocator (empty_cache is disabled), vLLM's weight/KV allocations are served from that cache (on a cache miss PyTorch
# frees cached blocks and immediately re-mallocs, a microsecond window), and vLLM's free-memory check is told about it.
_real_empty_cache = torch.cuda.empty_cache
torch.cuda.empty_cache = lambda: None
try:
    torch.accelerator.empty_cache = lambda: None
except Exception:
    pass
import vllm.v1.worker.gpu_worker as gw
RELEASE = {"n": 0}
class _Snap(gw.MemorySnapshot):
    def measure(self):
        k = RELEASE["n"]
        if k:
            for _ in range(min(k, len(PLACE))):
                PLACE.pop()
            RELEASE["n"] = 0
        super().measure()
        self.free_memory += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
gw.MemorySnapshot = _Snap
from vllm import LLM, SamplingParams

def regrab():
    while len(PLACE) < TOTAL and grab(0.5):
        pass
    log(f"re-grabbed -> holding {len(PLACE)}/{TOTAL} GB")

def run_model(key, limit=None):
    cfg = MODELS[key]
    out_path = RES / "gens" / (f"{key}.jsonl" if limit is None else f"{key}.test.jsonl")
    os.makedirs(out_path.parent, exist_ok=True)
    prompts = read_jsonl(DATA / "prompts.jsonl")
    if limit:
        import random; random.Random(0).shuffle(prompts); prompts = prompts[:limit]
    done = {r["pid"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [p for p in prompts if p["pid"] not in done]
    log(f"{key}: {len(done)} done, {len(todo)} to do")
    if not todo:
        return
    budget = min(len(PLACE), cfg["need"] + 6)      # extra KV for throughput when we have room
    kv = cfg["kv"] + max(0, budget - cfg["need"] - 1)
    RELEASE["n"] = budget
    llm = LLM(model=cfg["hf"], dtype="auto", max_model_len=2048, gpu_memory_utilization=(budget - 0.3) / 47.9,
              kv_cache_memory_bytes=int(kv * 2**30), enforce_eager=True, max_num_seqs=256, seed=SEED,
              trust_remote_code=False, enable_prefix_caching=True)
    sp = SamplingParams(temperature=0.0, max_tokens=256, seed=SEED)
    ctk = {"enable_thinking": False} if cfg.get("thinking_off") else None
    run = lambda convs: [o.outputs[0].text for o in llm.chat(convs, sp, use_tqdm=False, chat_template_kwargs=ctk)]
    single = [p for p in todo if len(p["turns"]) == 1]
    multi = [p for p in todo if len(p["turns"]) > 1]
    t0 = time.time()
    with open(out_path, "a") as f:
        for i in range(0, len(single), 1000):
            ch = single[i:i + 1000]
            for p, t in zip(ch, run([[{"role": "user", "content": p["turns"][0]}] for p in ch])):
                f.write(json.dumps(dict(pid=p["pid"], model=key, response=t, history=[]), ensure_ascii=False) + "\n")
            f.flush(); log(f"{key} {i+len(ch)}/{len(single)} single {time.time()-t0:.0f}s")
        if multi:
            convs = [[] for _ in multi]
            for r in range(max(len(p["turns"]) for p in multi)):
                for c, p in zip(convs, multi):
                    c.append({"role": "user", "content": p["turns"][r]})
                for c, t in zip(convs, run(convs)):
                    c.append({"role": "assistant", "content": t})
            for p, c in zip(multi, convs):
                h = [m["content"] for m in c if m["role"] == "assistant"]
                f.write(json.dumps(dict(pid=p["pid"], model=key, response=h[-1], history=h[:-1]), ensure_ascii=False) + "\n")
            log(f"{key} multi-turn done {time.time()-t0:.0f}s")
    del llm
    gc.collect()

def make_llm(budget_gb, **kw):
    """Create an in-process vLLM engine that may use `budget_gb` of the held memory (used by tasks.py)."""
    RELEASE["n"] = min(len(PLACE), budget_gb)
    return LLM(gpu_memory_utilization=(RELEASE["n"] - 0.3) / 47.9, enforce_eager=True, seed=SEED, **kw)

if __name__ == "__main__":
    qfile = sys.argv[1]
    finished = set()
    while not (LOGS / "STOP_worker").exists():
        queue = [l.strip() for l in open(qfile) if l.strip() and not l.startswith("#")]
        todo = [k for k in queue if k not in finished]
        if not todo:
            time.sleep(5); continue
        entry = todo[0]
        key, _, lim = entry.partition(":")
        try:
            if key.startswith("task="):            # dynamic tasks: tasks.py is re-imported every time -> code changes apply
                import importlib, tasks
                importlib.reload(tasks)
                getattr(tasks, key[5:])(sys.modules[__name__], lim)
            else:
                run_model(key, int(lim) if lim else None)
        except Exception as e:
            import traceback; traceback.print_exc(); log(f"FAILED {entry}: {e}")
        finished.add(entry)
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
            destroy_model_parallel(); destroy_distributed_environment()
        except Exception as e:
            log("destroy failed", e)
        gc.collect()
        regrab()
        log(f"FINISHED {entry}")
    log("STOP file seen; exiting")
