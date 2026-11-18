#!/usr/bin/env python3
"""Ask the writer for Hebrew terms (Hebrew script) with a one-line English context, by category.

  with-secrets uv run --with anthropic scripts/gen_terms.py data/terms/raw.jsonl [--per 50]
  -> appends {category, term, context, model, date}; resumes by category. Dedupe happens in build.
  --self: terms that describe the "I" and inflect for gender (2026-11-18 v4 top-up: v1-v3 copied held-out
  masculine adjective terms unchanged for a female "I", with only 134 inflected training examples).
"""
import argparse, datetime, json, os, re, sys

CATS = ["daily routine and household", "family and relationships", "work and the office", "health and medical care",
        "food, cooking and restaurants", "shopping, money and banking", "transport and driving", "IDF and reserve duty",
        "school, university and studying", "Jewish holidays, religion and tradition", "emotions and states of mind (adjectives)",
        "common everyday verbs (infinitive)", "phones, computers and the internet", "weather, nature and animals",
        "sports, hobbies and leisure", "government offices, bureaucracy and law", "travel and vacations",
        "media, culture and entertainment", "common Hebrew idioms and expressions", "the neighbourhood, community and city life"]
SELF_CATS = ["feelings and moods (adjectives)", "personality traits (adjectives)", "physical states: tired, hungry, sick, awake (adjectives and participles)",
             "social roles and statuses: renter, volunteer, newcomer, retiree (agent nouns)", "occupations and job titles",
             "lifestyle and identity: vegetarian, religious, secular, athlete", "relationship and family status: married, divorced, engaged",
             "attitudes and opinions: sure, opposed, convinced, worried (adjectives and participles)"]
SELF_PROMPT = """List {k} distinct modern Hebrew words or short expressions in the category "{cat}" that a person uses to describe themself ("I am ..."), in everyday Israeli speech.
Every item must have a feminine singular form whose SPELLING differs from the masculine (for example עייף/עייפה, שוכר/שוכרת, אסיר תודה/אסירת תודה). Skip words whose two forms are spelled the same (such as רוצה, יפה).
Give the masculine singular dictionary form, unpointed, and a short English context line that pins down the meaning.
Reply with a JSON array of objects {{"term": ..., "context": ...}} and nothing else."""
PROMPT = """List {k} distinct modern Hebrew terms in the category "{cat}" that an Israeli adult uses in everyday speech.
Mix nouns, verbs (infinitive), adjectives and short multi-word expressions where they fit the category.
Write each term in unpointed Hebrew script (no nikud), in its dictionary form, and give a short English context line that pins down the intended meaning.
Reply with a JSON array of objects {{"term": ..., "context": ...}} and nothing else."""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--per", type=int, default=50)
    ap.add_argument("--model", default="claude-sonnet-5-5")
    ap.add_argument("--self", action="store_true", help="gender-inflecting self-description terms (v4 top-up)")
    a = ap.parse_args()
    cats, prompt = (SELF_CATS, SELF_PROMPT) if a.self else (CATS, PROMPT)
    import anthropic
    c = anthropic.Anthropic()
    done = {json.loads(l)["category"] for l in open(a.out)} if os.path.exists(a.out) else set()
    with open(a.out, "a") as f:
        for cat in cats:
            if cat in done: continue
            m = c.messages.create(model=a.model, max_tokens=6000, thinking={"type": "between_tools"},
                                  messages=[{"role": "user", "content": prompt.format(k=a.per, cat=cat)}])
            raw = "".join(b.text for b in m.content if b.type == "text")
            items = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
            for it in items:
                f.write(json.dumps({"category": cat, **it, "model": a.model, "date": datetime.date.today().isoformat()},
                                   ensure_ascii=False) + "\n")
            print(cat, len(items), m.usage.input_tokens, m.usage.output_tokens, file=sys.stderr)


if __name__ == "__main__":
    main()
