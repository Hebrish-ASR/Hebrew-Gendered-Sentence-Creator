#!/usr/bin/env python3
"""Format gate for generated Hebrew sentences. Cheap, regex-only; the gender check is separate (analyser).

  python3 scripts/gate.py IN.jsonl   -> prints a summary, writes IN.gated.jsonl with per-sentence reasons

Checks: 6 to 18 words; the term exactly once (one-letter prefixes ובלמשכה attached are allowed, also
two, e.g. וב, שה; a multi-word term may take ה on its last word, ראיון העבודה); Hebrew letters, digits, spaces and basic punctuation only; no nikud.
"""
import json, re, sys

NIKUD = re.compile(r"[֑-ׇ]")
ALLOWED = re.compile(r"^[א-ת0-9\s.,!?;:\"'()\-–—%/״׳]+$")


def term_count(term, s):
    words = term.split()
    body = re.escape(term)
    if len(words) > 1:  # construct state takes the article on its last word: ראיון עבודה -> ראיון העבודה
        body = r"\s+".join(map(re.escape, words[:-1])) + r"\s+ה?" + re.escape(words[-1])
    pat = r"(?<![\u05D0-\u05EA])(?:[ובלמשכה]{0,2})" + body + r"(?![\u05D0-\u05EA])"
    return len(re.findall(pat, s))


def check(term, s):
    bad = []
    n = len(s.split())
    if not 6 <= n <= 18: bad.append(f"length:{n}")
    if NIKUD.search(s): bad.append("nikud")
    if not ALLOWED.match(s): bad.append("charset")
    c = term_count(term, s)
    if c != 1: bad.append(f"term_count:{c}")
    return bad


def main(path):
    out = path.replace(".jsonl", ".gated.jsonl")
    tot = ok = calls = calls_ok = 0
    reasons = {}
    with open(out, "w") as f:
        for l in open(path):
            r = json.loads(l); calls += 1
            sents = r.get("sentences") or []
            r["gate"] = [check(r["term"], s) for s in sents]
            for b in r["gate"]:
                tot += 1; ok += not b
                for x in b: reasons[x.split(":")[0]] = reasons.get(x.split(":")[0], 0) + 1
            calls_ok += bool(sents) and not any(r["gate"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"calls": calls, "calls_all_pass": calls_ok, "sentences": tot, "pass": ok, "reasons": reasons}))


if __name__ == "__main__":
    main(sys.argv[1])
