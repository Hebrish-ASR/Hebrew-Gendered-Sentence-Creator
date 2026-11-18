# Hebrew-Gendered-Sentence-Creator

![Status: planning](https://img.shields.io/badge/status-planning-blue) ![Stage: 1 of 2](https://img.shields.io/badge/pipeline-stage%201%20of%202-informational)

This model writes natural Hebrew sentences that use a given term in the sense its context describes, **for a speaker of a given gender**. It is the Hebrew counterpart of [`danielrosehill/Contextual-Sentence-Creator`](https://huggingface.co/danielrosehill/Contextual-Sentence-Creator), which does the same in English: a LoRA on `ibm-granite/granite-4.2-3b`, trained on `danielrosehill/term-context-sentences`.

Status (2026-11-18): planning. No data and no weights yet.

## The pipeline: why two models

Hebrew marks the speaker's gender in two different ways, and they need different tools.

1. **In the letters.** אני הולך / אני הולכת, אני שמח / אני שמחה, הייתי עייף / הייתי עייפה. Here the gender changes which letters are written, so the sentence has to be *written* for the right speaker. A nikud model cannot do this, because it only adds vowels to the letters it is given. **This repo (stage 1) handles it.**
2. **Only in the vowels.** אני רוצה is spelled the same for both genders but read רוֹצֶה (male) or רוֹצָה (female); the same goes for קונה, עושה, מנסה, יפה, קשה (present tense of ל"ה verbs and adjectives ending in ה). The letters are already right; only nikud can show the gender. **[Hebrew-Gendered-Nikud](https://github.com/Hebrish-ASR/Hebrew-Gendered-Nikud) (stage 2) handles it**: a fine-tune of `dicta-il/dictabert-large-char-menaked` with speaker gender as an input.

```
term + context + speaker_gender
        │
        ▼
[1] Hebrew-Gendered-Sentence-Creator  ──►  unpointed Hebrew, letters correct for the speaker
        │                                    (usable on its own: no-nikud output)
        ▼
[2] Hebrew-Gendered-Nikud (gender=m|f) ──►  pointed Hebrew, vowels correct for the speaker
```

Why not one model? Doing both would need a Hebrew LLM that writes reliable nikud, and that is exactly where LLMs are weak and dedicated nikud models are strong. With two models, each one can be tested and replaced on its own. The nikud model is also useful on any Hebrew text, not just text from stage 1.

Each run produces three outputs: unpointed, pointed for a male speaker and pointed for a female speaker. A sentence with nothing gender-specific comes out the same in both pointed versions.

## Prior work (searched 2026-11-18)

Nothing found for Hebrew generates text with speaker gender as an input. The closest template is Arabic:

- First-person gender identification and reinflection, [Habash et al. 2019](https://aclanthology.org/W19-3822/).
- The [Arabic Parallel Gender Corpus](https://aclanthology.org/2022.lrec-1.199): subtitle lines labelled male, female or ambiguous, with opposite-gender twins.
- [User-Centric Gender Rewriting](https://arxiv.org/abs/2205.02211) (2022): separate speaker and listener controls.

[MORPHOGEN](https://arxiv.org/abs/2604.18914) (2026) tests LLM first-person gender rewriting in French, Arabic and Hindi, and finds that models also flip the gender of *other* people in the sentence. Stage 1's data checks are built to catch that. The full survey is in the [Hebrew-Gendered-Nikud README](https://github.com/Hebrish-ASR/Hebrew-Gendered-Nikud#prior-work-searched-2026-11-18).

## Proposed interface

The same chat format as Contextual-Sentence-Creator, with Hebrew output and a `speaker_gender` field:

```json
{"term": "גמ\"ח", "context": "a free community lending fund", "n": 3, "speaker_gender": "f"}
```

The reply is a JSON array of `n` unpointed Hebrew sentences, each 6 to 18 words, each a statement in the first person where the term allows it. `speaker_gender` is `m`, `f` or `none` (no first-person gendered forms).

**What is conditioned:** the gender of the sentence's first person, the "I" (אני, or a first-person verb with no pronoun). That "I" is normally the speaker, which is where the field name comes from. Only first-person forms follow the setting. Every other person in the sentence keeps whatever gender the content needs.

## Planning notes (to be worked out)

- **Base model.** It must write good Hebrew at a size we can run cheaply. The shortlist and reasons are in [`docs/base-model-candidates.md`](docs/base-model-candidates.md). The working bet is `dicta-il/DictaLM-3.0-1.7B-Instruct`, against Gemma 4 E4B, DictaLM 2.0, DictaLM-3.0-Nemotron-12B and others. Not chosen until the phase-0 probe has run. The full proposal is in [`docs/training-plan.md`](docs/training-plan.md).
- **Data.** Distil from a strong LLM. Write each term's sentences as m/f pairs. Accept a pair only if the words that differ are the speaker's own forms (not "my wife" → "my husband"), checked by a cheap automatic diff.
- **Evaluation.** Gate pass rate (JSON, n, length, term present), speaker-gender agreement measured by a morphological analyser, and no gender flips on third parties.

## Licence

To be decided with the base model.
