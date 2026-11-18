#!/usr/bin/env python3
"""Ask the writer for Hebrew terms (Hebrew script) with a one-line English context, by category.

  with-secrets uv run --with anthropic scripts/gen_terms.py data/terms/raw.jsonl [--per 50]
  -> appends {category, term, context, model, date}; resumes by category. Dedupe happens in build.
"""
import argparse, datetime, json, os, re, sys

CATS = ["daily routine and household", "family and relationships", "work and the office", "health and medical care",
        "food, cooking and restaurants", "shopping, money and banking", "transport and driving", "IDF and reserve duty",
        "school, university and studying", "Jewish holidays, religion and tradition", "emotions and states of mind (adjectives)",
        "common everyday verbs (infinitive)", "phones, computers and the internet", "weather, nature and animals",
        "sports, hobbies and leisure", "government offices, bureaucracy and law", "travel and vacations",
        "media, culture and entertainment", "common Hebrew idioms and expressions", "the neighbourhood, community and city life"]
PROMPT = """List {k} distinct modern Hebrew terms in the category "{cat}" that an Israeli adult uses in everyday speech.
Mix nouns, verbs (infinitive), adjectives and short multi-word expressions where they fit the category.
Write each term in unpointed Hebrew script (no nikud), in its dictionary form, and give a short English context line that pins down the intended meaning.
Reply with a JSON array of objects {{"term": ..., "context": ...}} and nothing else."""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--per", type=int, default=50)
    ap.add_argument("--model", default="claude-sonnet-5-5"); a = ap.parse_args()
    import anthropic
    c = anthropic.Anthropic()
    done = {json.loads(l)["category"] for l in open(a.out)} if os.path.exists(a.out) else set()
    with open(a.out, "a") as f:
        for cat in CATS:
            if cat in done: continue
            m = c.messages.create(model=a.model, max_tokens=6000, thinking={"type": "between_tools"},
                                  messages=[{"role": "user", "content": PROMPT.format(k=a.per, cat=cat)}])
            raw = "".join(b.text for b in m.content if b.type == "text")
            items = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
            for it in items:
                f.write(json.dumps({"category": cat, **it, "model": a.model, "date": datetime.date.today().isoformat()},
                                   ensure_ascii=False) + "\n")
            print(cat, len(items), m.usage.input_tokens, m.usage.output_tokens, file=sys.stderr)


if __name__ == "__main__":
    main()
