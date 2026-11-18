#!/usr/bin/env python3
"""Distil gender-conditioned Hebrew sentences from a writer LLM (Anthropic API, sync).

Each call asks for n sentences for one (term, context) whose first person ("I") is a man (m) or a
woman (f), written natively in that gender. Nothing is written in one gender and edited into the other.

  with-secrets uv run --with anthropic scripts/gen_sentences.py TERMS.jsonl OUT.jsonl [--n 10] [--model M] [--batch]

--batch submits every pending (term, gender) as one Anthropic Message Batch (half price), polls until it
ends, and writes the results; the batch id is printed and stored in OUT.batch so a crash can resume.

TERMS.jsonl: {term, context}. OUT.jsonl (appends, resumes by term+gender):
{term, context, gender, model, sentences, raw, usage, date}. Gate results are added by scripts/gate.py.
"""
import argparse, datetime, json, os, re, sys

SYSTEM = """You write natural, everyday modern Hebrew sentences for a training dataset.

Rules:
- Write exactly {n} different sentences. Each is 6 to 18 words.
- Each sentence uses the term exactly once, spelled as given (a prefix letter such as ו, ה, ב, ל, מ, ש, כ may be attached), in the meaning the context describes.
- The sentences are in the first person. The "I" of every sentence is a {who}. Write every first-person form natively for a {who}: present-tense verbs, adjectives and participles that describe the "I", and any other form whose spelling depends on the gender of the "I".
- Only the "I" follows this rule. Other people in a sentence (you, he, she, my partner, my boss, my friend, my children) have whatever gender is natural; vary them and do not make them match the "I".
- Most sentences should contain at least one form whose spelling shows the gender of the "I". Do not force one where it would be unnatural.
- Vary openings, settings and sentence structure. Plain unpointed Hebrew letters: no nikud. Write numbers as digits.
- Reply with a JSON array of {n} strings and nothing else."""
WHO = {"m": "man", "f": "woman"}


def parse(raw):
    m = re.search(r"\[.*\]", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


def run_batch(client, a, terms, done):
    import time
    bfile = a.out + ".batch"
    keys = {}
    for i, t in enumerate(terms):
        for g in a.genders.split(","):
            if (t["term"], g) not in done:
                keys[f"r{i}-{g}"] = (t, g)
    if os.path.exists(bfile):
        bid = open(bfile).read().strip()
    else:
        reqs = [{"custom_id": k, "params": {"model": a.model, "max_tokens": 3000, "thinking": {"type": "between_tools"},
                 "system": SYSTEM.format(n=a.n, who=WHO[g]),
                 "messages": [{"role": "user", "content": json.dumps({"term": t["term"], "context": t["context"]}, ensure_ascii=False)}]}}
                for k, (t, g) in keys.items()]
        bid = client.messages.batches.create(requests=reqs).id
        open(bfile, "w").write(bid)
    print("batch", bid, len(keys), "requests", file=sys.stderr)
    while (b := client.messages.batches.retrieve(bid)).processing_status != "ended":
        print(b.request_counts, file=sys.stderr); time.sleep(60)
    tin = tout = 0
    with open(a.out, "a") as f:
        for res in client.messages.batches.results(bid):
            if res.custom_id not in keys or res.result.type != "succeeded":
                print("failed", res.custom_id, res.result.type, file=sys.stderr); continue
            t, g = keys[res.custom_id]; msg = res.result.message
            raw = "".join(b.text for b in msg.content if b.type == "text")
            tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
            f.write(json.dumps({**t, "gender": g, "model": a.model, "sentences": parse(raw), "raw": raw,
                                "usage": {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens},
                                "batch_id": bid, "date": datetime.date.today().isoformat()}, ensure_ascii=False) + "\n")
    print(json.dumps({"batch": bid, "tokens_in": tin, "tokens_out": tout}), file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("terms"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--genders", default="m,f")
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()
    import anthropic
    client = anthropic.Anthropic()
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["term"], r["gender"]))
    terms = [json.loads(l) for l in open(a.terms) if l.strip()]
    if a.batch:
        return run_batch(client, a, terms, done)
    with open(a.out, "a") as f:
        for t in terms:
            for g in a.genders.split(","):
                if (t["term"], g) in done:
                    continue
                who = WHO[g]
                msg = client.messages.create(
                    model=a.model, max_tokens=3000, thinking={"type": "between_tools"},
                    system=SYSTEM.format(n=a.n, who=who),
                    messages=[{"role": "user", "content": json.dumps({"term": t["term"], "context": t["context"]}, ensure_ascii=False)}])
                raw = "".join(b.text for b in msg.content if b.type == "text")
                m = re.search(r"\[.*\]", raw, re.S)
                try:
                    sents = json.loads(m.group(0)) if m else None
                except json.JSONDecodeError:
                    sents = None
                rec = {**t, "gender": g, "model": a.model, "sentences": sents, "raw": raw,
                       "usage": {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens},
                       "date": datetime.date.today().isoformat()}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                print(t["term"], g, len(sents or []), rec["usage"], file=sys.stderr)


if __name__ == "__main__":
    main()
