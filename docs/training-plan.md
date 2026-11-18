# Training plan (proposed)

Status 2026-11-18: **paused after v3 (Daniel).** v3 is no better than v2. Next step if resumed: the v4 targeted top-up described at the end. Earlier status: v2 trained (`danielrosehill/hebrew-gendered-sentence-creator-gemma-4-e4b-v2`, private). It is better than prompt-only Gemma on format and fluency, but not on gender correctness, because of how the term is handled. See the last Progress entry. Base-model candidates are in
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
2. **Generation, native per gender (no rewriting).** Each call asks the writer for n sentences for one term
   *as written by a male "I"* or *by a female "I"*, from scratch. A sentence is never written in one gender and
   then edited into the other, so the training targets look exactly like what the model must produce at
   runtime: one pass, gender as an input. The m and f batches for a term are independent sentences.
   Writer: pick by a small comparison (Claude Sonnet 5.5 vs Gemini 3.1 Pro / Flash on ~10 terms each,
   Daniel rates Hebrew naturalness). Cost estimate goes to Daniel before the bulk run.
3. **Automatic checks** (each rejection reason logged):
   - gate: words, term once, charset, no nikud, JSON;
   - **first-person agreement:** the analyser tags every first-person form (person=1, or subject אני /
     implied first person); any letter-marked form of the wrong gender rejects the sentence;
   - most sentences contain at least one letter-marked first-person form; a small share with none (only
     vowel-marked forms, or no gendered "I") is kept, so the model learns it need not force one in;
   - **third-party drift:** across the whole set, the gender of third persons (היא/הוא, אשתי/בעלי, their
     verbs) must not track the requested gender. Measured as a correlation; if it shows up, rebalance the
     prompt or filter (the MORPHOGEN failure, seen here as bias rather than as a flipped twin);
   - duplicate and overused-opening filter, as in the English pipeline.
   Analyser: `dicta-il/dictabert-joint` (or `dictabert-morph`) for person/gender/number per token. It is a
   BERT-base-sized model; run it on Modal for the bulk pass, not on the laptop.
4. **Examples.** Group by term. Each example samples n from 1 to min(available, 10) with a random order.
   Gender balance m:f 50:50, with the same terms on both sides.
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
- If fluent but gender errors remain: a DPO pass whose negatives are the sentences the analyser rejected
  for wrong first-person gender (and, if needed, a few minimal-edit opposite-gender copies made only for
  this purpose; they are never SFT targets).
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
`danielrosehill/`, as in the English sibling. Code is MIT; weights follow the base model (see the README's Licence section).

## Progress

### 2026-11-18: writer pilot, analyser, Gemma 4 smoke test

- **Writer pilot** (`scripts/gen_sentences.py`, `data/pilot/`): Claude Sonnet 5.5, 10 terms x m/f x n=10, native
  per-gender prompt. With adaptive thinking on, output tokens doubled and the prompt's example word (עייף) was
  parroted, so the first 7 calls were discarded (`sonnet-5-5-v0-prompt.jsonl`). Thinking is turned off with
  `thinking={"type": "between_tools"}`, because this model rejects `{"type": "disabled"}`. Final: 20 calls,
  8,320 tokens in and 8,967 out, about US$0.11 sync.
- **Gate** (`scripts/gate.py`): 196/200 pass after allowing the article on a multi-word term's last word
  (ראיון העבודה). Rejections were inflected forms (ראיונות, לחוץ).
- **Analyser** (`scripts/analyse_modal.py`, dictabert-joint on a Modal T4): the first version flagged 42
  "disagreements". All of them were tagger artefacts: first-person past/future verbs (קיבלתי) and pronouns
  (אני) tagged with a gender although their letters carry none, and vowel-only ל"ה forms (רוצה, מרוצה, מנסה).
  Those are now skipped. Result: m 69 agree / 2 disagree / 29 unmarked; f 62 / 7 / 31. All 9 remaining
  disagreements are vowel-only forms, fixed by the ה-lemma rule since. **The writer made no real gender
  errors in 200 sentences.** About 30% of sentences have no letter-marked first-person form.
- **Terms** (`scripts/gen_terms.py`): 20 categories x ~50 from Sonnet 5.5, giving 1,018 raw and 924 unique
  new terms in `data/terms/terms.jsonl`. About US$0.30.
- **Gemma 4 smoke test** (`scripts/train_modal.py --run gemma-smoke`, 64 examples, 4 steps, L40S):
  - The checkpoint loads as `Gemma4ForConditionalGeneration` with transformers 5.23.0. LoRA on the language
    model only gives 69.8M trainable parameters (0.87%).
  - Batch-16 logits ran out of memory (262k vocabulary), so training uses micro-batches of 4 with gradient
    accumulation.
  - About 2.75 s per step.
  - After only 4 steps the output is valid JSON with mostly correct gender, but the Hebrew has errors
    (הראיון עבודה, האשת חברתי) and there was one m/f slip (אני צריכה in an m batch).
  - The turn-end token is `<turn|>` (the smoke run appended `<eos>`; fixed).
- Pricing note: the English sibling's logged batch run implies Sonnet 5.5 at US$1/US$5 per M tokens in
  batch, so US$2/US$10 sync.

### 2026-11-18: does prompting alone do it? Yes

Daniel asked whether a system prompt to Gemma could replace the fine-tune. Test:
`scripts/train_modal.py --run gemma-4-e4b-zero-shot --zero-shot`, which uses `google/gemma-4-E4B-it` with no
training and the writer's full prompt from `gen_sentences.py`, on the same 10 pilot terms x m/f x n=10. Output
in `data/runs/gemma-4-e4b-zero-shot/`. Both runs were scored with the same gate and analyser (final rules).

| | Gemma 4 E4B, prompt only | Claude Sonnet 5.5, prompt only |
|---|---|---|
| JSON parses, exactly n | 20/20 | 20/20 |
| gate pass | 195/200 | 196/200 |
| first-person gender **disagree** | **0** | **0** |
| letter-marked first-person form present | 96/200 | 121/200 |

Reading the Gemma output (Daniel's assistant, not Daniel): the gender is right throughout. The Hebrew is
mostly correct but plainer, many sentences open with "אני", and there are occasional errors: אני נסה,
מרגיש לי לחץ, ממלא את המילואים, המורכבות ביותר.

**Conclusion.** Gender conditioning does not need a fine-tune, because a prompt does it, even on a 4B-effective
model. A fine-tune makes no model deterministic. The guarantee comes from the analyser check, which rejects
and retries any sentence with a wrong-gender letter-marked first-person form, and that wraps any model. The
LoRA would only buy Hebrew fluency and variety in a small local model, by distilling from Sonnet. Resume the
bulk run (planned above: 924 terms, Sonnet batch about US$4.90, analyser about US$0.10, E4B LoRA about
US$1.50) only if that need comes up.

Spend for the whole investigation:
- Anthropic: about US$0.52 (pilot v0 + v1 + terms, sync).
- Modal: two smoke runs, one zero-shot run and three analyser runs, roughly US$0.50 (estimated, not read
  from the bill).

### 2026-11-18: full run, v1 and v2 (Daniel: "let's do the fine-tune")

**Data.**
- Anthropic batch `msgbatch_01PoCafFGsoKAuBRB1JkazQ8`: 924 terms x m/f x n=10, so 1,848 requests.
  771,430 tokens in and 847,589 out, US$5.01.
- 6 calls did not parse. The analyser flagged 82 sentences, almost all female "I" with masculine forms
  (אני חרד, הרגשתי נרגש), so even Sonnet slips about 1% of the time on female speakers.
- Dataset: 834 train terms and 100 eval terms (seed 0). v1 had 6,748 examples; v2 has 6,884.

**v1 bug, fixed in v2.** `gate.py` required the term exactly as given. For a female "I", adjective terms must
inflect (אסיר תודה → אסירת תודה, מדוכא → מדוכאה); the writer did that, and the gate threw those sentences out.
The female data for adjective terms was therefore left with masculine uses only. In v1's eval, 39 of the 41
female gender errors were the term itself. The gate now accepts gender/number inflection of the term.

**Runs** (L40S, LoRA r=32, lr 2e-4, 2 epochs, micro-batch 4 x 4):
- v1: 844 steps, 1,828 s.
- v2: 861 steps, 2,237 s.
- v2's first attempt finished training, but its local client was killed by laptop memory pressure and the
  app was stopped before eval and push. Runs now use `modal run --detach` and write to the Modal volume
  `hebrew-gendered-runs`.

**Eval** (100 held-out terms x m/f, n=10, temperature 0.8; the same gate and analyser for all three):

| | Gemma E4B prompt-only (writer prompt) | v1 | v2 |
|---|---|---|---|
| unparseable / not exactly n | 6 / 6 | 0 / 0 | 1 / 1 |
| gate pass | 1,855/1,940 | 1,956/2,000 | 1,921/1,990 |
| female "I": marked forms agree / disagree | 510 / 19 (3.6%) | 668 / 41 (5.8%) | 680 / 26 (3.7%) |
| male "I": agree / disagree | 488 / 2 | 773 / 3 | 740 / **0** |
| sentences with a letter-marked "I" form | 1,019 | 1,485 | 1,446 |

**Reading v2's 26 female disagreements:**
- 15 are the term kept in the masculine (מדוכא x10, שוכר x5). Both are held-out terms; the system prompt
  still says "use the term once, as given".
- 5 are genuine slips: ער x3, גר, מתחיל.
- 6 are analyser false positives: נוח, טוב, חוקי and פתוח used as predicates of other things, מת, נמאס.

So the only real weakness left is an adjective term for a woman, and the cause is the prompt wording. The
male side is clean.

**Fluency**, read by Claude rather than Daniel:
- v2 is more natural and varied than prompt-only (for example "כשאני לבד בבית, ארוחת ערב שלי היא בדרך כלל משהו שאני מחממת מהמקפיא").
- Prompt-only output is flat and repeats the pronoun (אני הכנתי, אני קניתי).
- Both still make content errors: v2 wrote "האטיג" and put the etrog on Yom Kippur.

**Next (v3, needs approval):**
1. Change `scripts/system_prompt.py` and the writer prompt so the term "may be inflected to agree with
   the 'I'".
2. Rebuild, which needs no new data.
3. Retrain detached, about US$1.50.
Expected effect: the term-related female errors (15 of the 26) go away.

**Spend to date**, Modal figures estimated from run times at about US$0.00054/s for an L40S, not read from
the bill:
- Anthropic: US$5.53.
- Modal: about US$5.30 (v1, v2 x2, zero-shot x2, smoke tests, analysers).
- Total about US$10.80, against the US$6.50 approved. The overrun is the v1 gate bug (one retrain), the lost
  v2 client (one more retrain) and the like-for-like zero-shot baseline.

### 2026-11-18: v3, and pause

v3 used the same data as v2. The only change is that the system prompt (`scripts/system_prompt.py`) and the
writer prompt now say the term may be inflected to agree with the "I". v3 is
`danielrosehill/hebrew-gendered-sentence-creator-gemma-4-e4b-v3` (private): 861 steps, 1,935 s training,
346 s eval. On the same 100 eval terms:

| | prompt-only | v2 | v3 |
|---|---|---|---|
| unparseable | 6 | 1 | 3 |
| female "I" disagree | 19 | 26 | 22 |
| male "I" disagree | 2 | 0 | 2 |

**v3 = v2 within noise.** 17 of v3's 22 female errors are the held-out term copied in the masculine (אסיר
תודה x8, מדוכא x6, שוכר x5 and צמחוני; some sentences hold more than one). A prompt change does not reach
this failure, because the model saw only 134 training examples of an inflected term.

**v4 plan (paused by Daniel on 2026-11-18, not run):**
1. `scripts/gen_terms.py --self` asks for about 300 self-description terms whose feminine form is spelled
   differently: moods, traits, physical states, roles, occupations, identities, family status, attitudes.
   The run was stopped part-way; partial output may be in `data/terms/self-raw.jsonl`.
2. Generate m/f sentences for those terms with `gen_sentences.py --batch` (about US$1).
3. Rebuild with `build_dataset.py --eval-from data/eval-terms-fixed.jsonl`, which keeps the same 100 eval
   terms so v4 compares with v2 and v3; drop any new term that is an eval term.
4. Retrain detached (about US$1.50) and score.

Ship any version with a verified wrapper: generate, run the analyser, regenerate rejects.

Workaround for callers today: pass the term already inflected for the speaker (מדוכאה for f).

Not published: Hugging Face publishing (`scripts/publish_modal.py`, targets `Hebrish/Hebrew-Gendered-Sentence-Creator`
and `Hebrish/Hebrew-Gendered-Sentences`; cards in `hf/` not yet written) waits for Daniel.

Spend to date: about US$13.80 (Anthropic US$5.53, Modal about US$8.30 estimated from run times).
