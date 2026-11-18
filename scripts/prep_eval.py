#!/usr/bin/env python3
"""Turn a run's eval-outputs.jsonl into gate/analyser input: data/runs/<run>/eval.jsonl.

  python3 scripts/prep_eval.py RUN
  python3 scripts/gate.py data/runs/RUN/eval.jsonl
  MODAL_PROFILE=danielrosehill modal run scripts/analyse_modal.py --path data/runs/RUN/eval.gated.jsonl
"""
import json, re, sys
from pathlib import Path

d = Path(__file__).resolve().parents[1] / "data/runs" / sys.argv[1]
bad = n_ok = 0
with open(d / "eval.jsonl", "w") as f:
    for l in open(d / "eval-outputs.jsonl"):
        r = json.loads(l)
        m = re.search(r"\[.*\]", r["raw"], re.S)
        try:
            s = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            s = None
        if not isinstance(s, list) or not all(isinstance(x, str) for x in s):
            bad += 1; s = []
        n_ok += len(s) == r["n"]
        f.write(json.dumps({"term": r["term"], "context": r["context"], "gender": r["gender"], "sentences": s},
                           ensure_ascii=False) + "\n")
print(json.dumps({"unparseable": bad, "exactly_n": n_ok}))
