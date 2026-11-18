#!/usr/bin/env python3
"""Format gate for generated Hebrew sentences. Cheap, regex-only; the gender check is separate (analyser).

  python3 scripts/gate.py IN.jsonl   -> prints a summary, writes IN.gated.jsonl with per-sentence reasons

Checks: 6 to 18 words; the term exactly once (one-letter prefixes ובלמשכה attached are allowed, also
two, e.g. וב, שה; a multi-word term may take ה on its last word, ראיון העבודה); Hebrew letters, digits, spaces and basic punctuation only; no nikud.
"""
import json, re, sys

NIKUD = re.compile(r"[֑-ׇ]")
ALLOWED = re.compile(r"^[א-ת0-9\s.,!?;:\"'()\-–—%/״׳]+$")


FEM = r"(?:ה|ת|ית|את|אה|ות|ים)?"  # gender/number inflection of the term (מדוכא -> מדוכאה, אסיר -> אסירת, קנאי -> קנאית)


def term_body(words):
    w = [re.escape(x) for x in words]
    if len(w) == 1:
        if words[0].endswith("ה"):  # מרוצה, שמחה: ה -> ת (מרוצת) is rare; allow the plain form
            return w[0]
        return w[0] + FEM
    # multi-word: the first word may inflect (construct אסירת תודה); the last may take the article
    return r"\s+".join([w[0] + FEM] + w[1:-1] + [r"ה?" + w[-1]])


def term_count(term, s):
    """The term once, with up to two one-letter prefixes, and with a gender/number inflection allowed.
    History (2026-11-18): an exact-form gate rejected אסירת תודה / מדוכאה for female speakers, so the
    female training data kept only masculine uses of adjective terms, and gemma-4-e4b-v1 learned to write
    'אני אסיר תודה' for a woman (39 of its 41 female gender errors)."""
    pat = r"(?<![\u05D0-\u05EA])(?:[ובלמשכה]{0,2})" + term_body(term.split()) + r"(?![\u05D0-\u05EA])"
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
