"""Shared paths, trigger-word lexicon and helpers."""
import json, os, re, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RES = ROOT / "results"
FIG = ROOT / "figures"
SEED = 42

# A-priori "safety trigger" lexicon: surface words that are typical refusal triggers in safety-tuning data.
# Fixed BEFORE looking at any model output. Matched case-insensitively as whole words with simple inflections (kill/kills/killing/killer).
TRIGGER_WORDS = """
kill murder assassinate suicide self-harm harm hurt injure attack assault violence violent weapon gun firearm
bomb explosive explode detonate grenade terror terrorist shoot shooting stab poison toxin lethal deadly
weaponize bioweapon nerve sarin anthrax ricin meth methamphetamine cocaine heroin fentanyl drug narcotic
hack hacking hacker malware ransomware virus keylogger exploit phishing ddos botnet trojan spyware crack
steal theft rob robbery fraud scam counterfeit launder laundering embezzle illegal illegally crime criminal
smuggle traffick trafficking kidnap abduct hostage torture abuse rape sexual porn pornographic explicit nude child
minor racist racism hate slur nazi genocide extremist radicalize propaganda harass harassment bully stalk
blackmail extort extortion threaten threat defamatory defame libel disinformation misinformation fake
cheat evade evasion bypass jailbreak dangerous unauthorized pirate piracy dox doxx arson
""".split()
_TRIG_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(set(TRIGGER_WORDS), key=len, reverse=True)) + r")(?:s|es|ed|d|ing|er|ers|ly|ally)?\b", re.I)

def trigger_hits(text: str):
    return [m.group(1).lower() for m in _TRIG_RE.finditer(text or "")]

def trigger_count(text: str) -> int:
    return len(trigger_hits(text))

def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
