---
license: mit
language:
- he
task_categories:
- text-generation
tags:
- hebrew
- gender
- synthetic
- tts
pretty_name: Hebrew Gendered Sentences
size_categories:
- 1K<n<10K
---

# Hebrew-Gendered-Sentences (work in progress)

These are synthetic first-person Hebrew sentences, written natively for a male or a female "I". Each sentence
uses a given term in the sense a context line describes. The dataset is the training data for
[`Hebrish/Hebrew-Gendered-Sentence-Creator`](https://huggingface.co/Hebrish/Hebrew-Gendered-Sentence-Creator).
Code and the experiment log are in
[Hebrish-ASR/Hebrew-Gendered-Sentence-Creator](https://github.com/Hebrish-ASR/Hebrew-Gendered-Sentence-Creator).

## Files

- **`train.jsonl`:** 6,884 chat examples, `{"messages": [system, user, assistant]}`.
  - The user turn is `{"term", "context", "n", "speaker_gender"}`.
  - The assistant turn is a JSON array of `n` sentences.
  - Split: 3,431 `f` and 3,453 `m` examples over 834 terms.
- **`eval.jsonl`:** the 100 held-out terms with contexts. No eval term appears in train.
- **`terms.jsonl`:** the 924 generated terms, `{term, context, category}`, in 20 everyday categories. Ten
  further pilot terms were hand-picked.
- **`build.json`:** build provenance and counts.

## How it was made (2026-11-18)

1. **Terms.** Claude Sonnet 5.5 listed about 50 everyday Hebrew terms per category, each with an English
   context line. After deduplication there are 924.
2. **Sentences.** For each term and each gender, Sonnet 5.5 wrote 10 sentences of 6 to 18 words, written
   directly for a man or a woman as the "I". No sentence was converted from the other gender. Generation was
   one Anthropic batch, `msgbatch_01PoCafFGsoKAuBRB1JkazQ8`.
3. **Filters:**
   - a format gate: 6 to 18 words, the term once (prefixes and gender/number inflection allowed), Hebrew
     letters only, no nikud;
   - a first-person gender check with [`dicta-il/dictabert-joint`](https://huggingface.co/dicta-il/dictabert-joint):
     any letter-marked first-person form of the wrong gender rejects the sentence. That was 79 sentences,
     nearly all masculine forms written for a female "I".
   - In total 17,725 of 18,632 sentences were kept.
4. **Examples.** Sentences are grouped by (term, gender). Each example samples `n` between 1 and 10, so each
   sentence is used about twice.

## Caveats

- The sentences are synthetic, written by an LLM. No human has reviewed them all; spot checks found them
  natural.
- About 30% of sentences have no letter-marked first-person form (for example, only past-tense verbs, which
  are not gendered in the first person). They are valid under either gender.
- Few terms are self-description adjectives that must inflect for a female speaker. This is the model's main
  weakness, and a top-up is planned.

## Licence

MIT.
