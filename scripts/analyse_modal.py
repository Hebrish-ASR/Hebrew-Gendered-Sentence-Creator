"""First-person gender check with dicta-il/dictabert-joint, on Modal (never on the laptop).

    MODAL_PROFILE=danielrosehill modal run scripts/analyse_modal.py --path data/pilot/sonnet-5-5.gated.jsonl

Writes <path minus .jsonl>.analysed.jsonl. For each sentence: the first-person gendered forms found, their
genders, and a verdict. A form counts as first person when (a) its own features say Person=1 and carry a
gender, or (b) it heads an `nsubj` child אני, or (c) it heads a `cop` child with Person=1 (הייתי עייפה).
Verdict: `agree` (all such forms match the requested gender), `disagree` (any form has the other gender),
`unmarked` (no letter-marked first-person form found). The rules are heuristics: check disagreements by eye.
"""
import json
from pathlib import Path

import modal

MODEL = "dicta-il/dictabert-joint"
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.7.1", "transformers==4.57.1")
app = modal.App("hebrew-gendered-analyse", image=image)
G = {"m": "Masc", "f": "Fem"}


@app.function(gpu="T4", timeout=1800)
def analyse(sentences: list[str]) -> list:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).cuda().eval()
    out = []
    for i in range(0, len(sentences), 32):
        with torch.no_grad():
            out += model.predict(sentences[i:i + 32], tok, output_style="json")
    return out


def first_person_forms(parsed):
    """Letter-marked gendered forms that belong to the "I". First-person past/future verbs (קיבלתי, אשחרר)
    and pronouns (אני, לי) are not gender-marked in Hebrew, even when the tagger assigns them a gender,
    so they are skipped; so are plurals (we). Predicates of אני or of a first-person copula count only
    when they are verbs or adjectives (the 2026-11-18 pilot showed nouns such as חצי misfiring)."""
    toks = parsed["tokens"]
    found = []
    for i, t in enumerate(toks):
        m = t.get("morph") or {}
        feats, pos = m.get("feats") or {}, m.get("pos")
        g = feats.get("Gender")
        if g not in ("Masc", "Fem") or pos in ("PRON", "DET", "ADP") or feats.get("Number") == "Plur":
            continue
        if feats.get("Person") == "1" and feats.get("Tense") in ("Past", "Fut"):
            continue
        if t["token"].endswith("ה") and str(t.get("lex") or "").endswith("ה"):
            continue  # vowel-only gender (רוצה, מרוצה, מנסה): the letters cannot show it; stage 2's job
        why = "person1" if feats.get("Person") == "1" else None
        if pos in ("VERB", "ADJ", "AUX"):
            for c in toks:
                syn = c.get("syntax") or {}
                if syn.get("dep_head_idx") != i:
                    continue
                cf = (c.get("morph") or {}).get("feats") or {}
                if syn.get("dep_func") == "nsubj" and c["token"] in ("אני", "ואני", "שאני", "כשאני"):
                    why = why or "nsubj_ani"
                if syn.get("dep_func") == "cop" and cf.get("Person") == "1":
                    why = why or "cop_person1"
            syn = t.get("syntax") or {}
            h = syn.get("dep_head_idx")
            if pos == "ADJ" and syn.get("dep_func") == "xcomp" and h is not None and 0 <= h < len(toks):
                if ((toks[h].get("morph") or {}).get("feats") or {}).get("Person") == "1":
                    why = why or "xcomp_of_person1"  # הרגשתי עייף
        if why:
            found.append({"token": t["token"], "gender": g, "why": why, "pos": pos, "feats": feats})
    return found


@app.local_entrypoint()
def main(path: str):
    rows = [json.loads(l) for l in open(path)]
    flat = [s for r in rows for s in (r.get("sentences") or [])]
    parsed = analyse.remote(flat)
    Path(path.replace('.jsonl', '.parsed.json')).write_text(json.dumps(parsed, ensure_ascii=False))
    k = 0
    for r in rows:
        r["analysis"] = []
        for s in r.get("sentences") or []:
            forms = first_person_forms(parsed[k]); k += 1
            gs = {f["gender"] for f in forms}
            verdict = "unmarked" if not gs else ("agree" if gs == {G[r["gender"]]} else "disagree")
            r["analysis"].append({"forms": forms, "verdict": verdict})
    out = path.replace(".jsonl", ".analysed.jsonl")
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    from collections import Counter
    c = Counter((r["gender"], a["verdict"]) for r in rows for a in r["analysis"])
    print(json.dumps({f"{g}:{v}": n for (g, v), n in sorted(c.items())}))
