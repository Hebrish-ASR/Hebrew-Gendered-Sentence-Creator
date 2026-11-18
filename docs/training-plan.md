# Training plan (proposed)

Status 2026-11-18: proposal only. Nothing has been generated, trained or spent. Base-model candidates are in
[`base-model-candidates.md`](base-model-candidates.md).

The template is the English sibling, `~/repos/github/ai-models/Term-Context-Sentence-Generator`
(HF `danielrosehill/Contextual-Sentence-Creator`): LoRA r=32 on all linear layers, lr 2e-4, 2 epochs,
batch 16, 13,408 examples, one Modal L40S, 1,622 s, about US$1.10 per run (`docs/plan.md`, `scripts/train_modal.py`
there). Its first run learned the JSON format perfectly (309/309) but missed the gate on length and term count
(95.7% of sentences vs Flash-Lite's 99.7%). Expect the same shape of problem here, plus Hebrew fluency and gender.

## Scope (Daniel, 2026-11-18)

- **Hebrew only.** Terms are written in Hebrew script, and output is Hebrew. No romanised Hebrish anywhere: not
  as input, not in the data, not in the eval.
- **Gender is a required, deterministic switch:** `m` or `f`, with no `none` and no default. Whoever calls the
  model always knows who is speaking.

Illustrative use case: **Hebrew podcast generation for TTS.** A script has a male and a female character. Each
character's lines must already be in the right gender in the text, before the TTS stage, because no TTS voice can
fix הלכתי עייף into הלכתי עייפה. The caller passes `m` for one character and `f` for the other, and gets lines
whose first-person forms are spelled for that character. Stage 2 (Hebrew-Gendered-Nikud) then gets the
vowel-only forms (רוֹצֶה / רוֹצָה) right for the TTS.

## Task contract

**The core of the task:** gender-conditioned generation from the point of view of the sentence's first person,
the "I" (אני, and first-person verb forms with no pronoun). In a read-aloud or dictation sentence that "I" is
usually the speaker, hence the field name `speaker_gender`, but the condition is defined on the first person
in the text, not on whoever reads it. Only first-person forms are conditioned. Second and third persons (את/אתה,
היא/הוא, אשתי, החברה שלי) are not, and must not drift toward the requested gender.

Input (chat user turn, JSON):

```json
{"term": "גמ\"ח", "context": "a free community lending fund", "n": 3, "speaker_gender": "f"}
```

Output: a JSON array of exactly `n` unpointed Hebrew strings, nothing else.

- 6 to 18 words; the term exactly once, as given (prefixes ו/ה/ב/ל/מ/ש/כ attached are allowed and must be
  matched as such); Hebrew letters, digits and basic punctuation only; **no nikud**.
- `speaker_gender` = `m` | `f`: at least one sentence in the batch (target: most) has a first-person form whose
  letters show the speaker's gender (past/future/present verbs and adjectives that differ in spelling:
  הולך/הולכת, שמח/שמחה, עייף/עייפה, מוכן/מוכנה). Every first-person form agrees with the speaker.
- A sentence whose first-person forms are all vowel-only (רוצה) or that has no gendered first-person form is
  still valid under either setting, but it should be the minority in a batch.
- Other people in the sentence keep whatever gender the content needs; the speaker setting never touches them.
- Forms whose gender is only in the vowels (רוצה, קונה, יפה) are allowed under any setting: stage 2
  (Hebrew-Gendered-Nikud) resolves them.

## Phase 0: base-model selection (wide and shallow)

1. **Tokenizer check, free.** For each candidate, tokenise a fixed set of ~200 Hebrew sentences and record
   tokens per word. Tokenizer files only (no weights), so it is fine on the laptop. Above ~3 tokens per word
   costs speed and context; note it, do not disqualify on it alone.
2. **Zero-shot probe, under US$0.25 total.** The same prompt and gate for every candidate: 8 terms x 2 genders
   (m, f) x 1 call, n=5, so 16 calls and 80 sentences per model. Through OpenRouter where the model is listed;
   otherwise one Modal vLLM container per model (an L4 for ~10 minutes is cents). Record: JSON parses, exactly
   n, gate pass, gender agreement (automatic, see Evaluation), and a 1-to-5 Hebrew fluency read by Daniel on
   10 sentences per model. Raw outputs to `data/zero-shot-bench/<date>.jsonl`.
3. **Pilot LoRA on the top 2 or 3**, with the phase-1 pilot data (~2k examples), 1 epoch, on one L40S.
   Roughly US$0.20 to 0.50 each: **needs Daniel's yes** on the list and estimate before running.
4. Pick the smallest model that is fluent and agrees on gender. Fluency is what fine-tuning fixes least,
   so it outranks JSON discipline in the choice.

## Phase 1: data

Distil from a strong LLM, then filter hard. Target ~15k training examples (the English run used 13.4k).

1. **Terms.** Hebrew-script terms only, each with a short context line. Sources to check: a general Hebrew
   frequency list, and Hebrew vocabulary lists (nouns, verbs, adjectives, expressions, names, places). No
   romanised terms; the English sibling's romanised term list is not a source.
2. **Generation, as pairs.** One call per term returns k sentence pairs: a male-speaker sentence and its
   female-speaker twin that differs only in the forms of the first person.
   Writer: pick by a small comparison (Claude Sonnet 5.5 vs Gemini 3.1 Pro / Flash on ~10 terms each,
   Daniel rates Hebrew naturalness). Cost estimate goes to Daniel before the bulk run.
3. **Automatic checks** (each rejection reason logged):
   - gate: words, term once, charset, no nikud, JSON;
   - **pair diff:** align m and f word by word; every differing token must be a first-person form of the
     speaker (analyser says person=1 or subject is אני / implied first person, gender m vs f). A difference
     anywhere else (אשתי → בעלי, third-party verbs) rejects the pair: this is the MORPHOGEN flip;
   - most pairs contain at least one letter-marked first-person form (identical twins are kept only as a
     small share of gender-neutral examples, so the model learns it need not force a form into every sentence);
   - duplicate and overused-opening filter, as in the English pipeline.
   Analyser: `dicta-il/dictabert-joint` (or `dictabert-morph`) for person/gender/number per token. It is a
   BERT-base-sized model; run it on Modal for the bulk pass, not on the laptop.
4. **Examples.** Group by term. Each example samples n from 1 to min(available, 10) with a random order.
   Gender balance m:f 50:50. Both twins of a pair may appear, in different examples, so the
   model sees minimal contrasts.
5. **Split by term:** ~300 held-out terms, stratified by part of speech. Plus a fixed bench of 8 to 10 terms
   used in every comparison.
6. Human spot check before training: Daniel reads ~50 pairs.

Outputs: `data/train.jsonl`, `data/eval.jsonl`, `data/build.json` (provenance, counts, rejection reasons).

## Phase 2: fine-tune

Port `Term-Context-Sentence-Generator/scripts/train_modal.py` to `scripts/train_modal.py` here.

- LoRA r=32, alpha=32, dropout 0.05, all linear layers; loss on the assistant turn only.
- lr 2e-4, 2 epochs, batch 16, length-bucketed batches; one Modal L40S.
- The shortlisted tokenizers need about 1.8 to 2.5 tokens per Hebrew word. The English run used Granite,
  which needs 5.4 tokens per Hebrew word. Estimate US$0.50 to 1.50 per full run on the 1.7B model and
  US$2 to 5 on 7 to 12B. Confirm with Daniel before the first full run.
- Merge and push to private HF `danielrosehill/hebrew-gendered-sentence-creator-<run>`.
- Run records in `data/runs/<run>/` (`train.json`, `eval-outputs.jsonl`, `score.json`).
- If fluent but gender errors remain: a DPO pass with the rejected opposite-gender twin as the negative.
  The pair data gives these negatives for free.

## Evaluation (go / no-go)

On the held-out terms and the bench, n=10, temperature 0.8, against the distillation writer zero-shot (quality
ceiling) and a cheap hosted model zero-shot (the price to beat):

1. **Format:** JSON parses, exactly n, gate pass rate, share of calls with all n passing.
2. **Speaker agreement:** share of `m`/`f` sentences whose speaker forms all match the requested gender
   (analyser), and share with at least one letter-marked speaker form.
3. **No third-party flips:** generate the same term with m and f at a fixed seed; flag gendered forms of
   non-speakers that differ between the two. Also an analyser count of third-person forms whose gender
   matches the speaker more often than chance.
4. **Fluency and sense:** an LLM judge scores 1 to 5 (natural Hebrew, term in the given sense), and Daniel
   reads a sample. The judge is a filter, not the verdict.
5. **Cost and latency** per 10 sentences on GPU and on local CPU (GGUF Q4_K_M via llama.cpp).

Ship when 1 and 2 are at the cheap baseline's level, 3 shows no systematic flips, and Daniel's read of 4
is "natural".

## Publishing (Daniel, 2026-11-18)

When the model passes evaluation, publish it under Hebrish. The code is already in the GitHub org `Hebrish-ASR`
(`Hebrish-ASR/Hebrew-Gendered-Sentence-Creator`). The weights and dataset go public on Hugging Face under the
Hebrish account; its exact HF org name has not been checked yet. Until then, runs push privately to
`danielrosehill/`, as in the English sibling. The release licence follows the base model.

## Open questions for Daniel

- Licence follows the base model; Apache-2.0 bases keep the options widest.
