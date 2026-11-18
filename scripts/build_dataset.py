#!/usr/bin/env python3
"""Build data/train.jsonl and data/eval.jsonl from analysed writer output.

  python3 scripts/build_dataset.py IN.analysed.jsonl [IN2 ...] [--eval-terms 300] [--seed 0]

Keeps a sentence when it passes the gate (scripts/gate.py) and the analyser verdict is not `disagree`
(scripts/analyse_modal.py). Groups by (term, gender); each kept group yields examples whose n is drawn from
1 to min(available, 10), each a different subset and order, until every sentence has been used about twice.
The split is by term: eval terms never appear in train. Writes data/build.json with counts.
"""
import argparse, collections, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from system_prompt import SYSTEM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--eval-terms", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uses", type=float, default=2.0)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    groups, ctx, drop = collections.defaultdict(list), {}, collections.Counter()
    for p in a.inputs:
        for l in open(p):
            r = json.loads(l)
            ctx[r["term"]] = r["context"]
            for s, g, an in zip(r.get("sentences") or [], r["gate"], r["analysis"]):
                if g: drop["gate"] += 1; continue
                if an["verdict"] == "disagree": drop["gender"] += 1; continue
                groups[(r["term"], r["gender"])].append(s)
    terms = sorted(ctx); rnd.shuffle(terms)
    ev_terms = set(terms[:a.eval_terms])
    train = []
    for (term, g), sents in sorted(groups.items()):
        if term in ev_terms: continue
        budget = int(len(sents) * a.uses)
        while budget > 0:
            n = rnd.randint(1, min(len(sents), 10))
            pick = rnd.sample(sents, n)
            user = json.dumps({"term": term, "context": ctx[term], "n": n, "speaker_gender": g}, ensure_ascii=False)
            train.append({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user},
                                       {"role": "assistant", "content": json.dumps(pick, ensure_ascii=False)}]})
            budget -= n
    rnd.shuffle(train)
    (ROOT / "data/train.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in train))
    ev = [{"term": t, "context": ctx[t]} for t in sorted(ev_terms)]
    (ROOT / "data/eval.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ev))
    kept = sum(len(v) for k, v in groups.items())
    info = {"inputs": a.inputs, "sentences_kept": kept, "dropped": dict(drop), "terms": len(terms),
            "eval_terms": len(ev_terms), "train_examples": len(train),
            "by_gender": dict(collections.Counter(json.loads(x["messages"][1]["content"])["speaker_gender"] for x in train))}
    (ROOT / "data/build.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
