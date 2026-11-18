---
license: apache-2.0
base_model: google/gemma-4-E4B-it
language:
- he
library_name: transformers
pipeline_tag: text-generation
datasets:
- Hebrish/Hebrew-Gendered-Sentences
tags:
- hebrew
- gender
- text-generation
- tts
- lora
- work-in-progress
---

# Hebrew-Gendered-Sentence-Creator (work in progress, v3)

> **Status: work in progress (2026-11-18).** It is usable, with a known weakness described under
> [Limitations](#limitations). Code, training scripts and the full experiment log are in
> [Hebrish-ASR/Hebrew-Gendered-Sentence-Creator](https://github.com/Hebrish-ASR/Hebrew-Gendered-Sentence-Creator).

The model writes natural unpointed Hebrew sentences that use a given term in the sense its context describes,
**with the gender of the sentence's first person (the "I", usually the speaker) as an input**. Each sentence
is written in one pass, directly in that gender. Existing text is not rewritten.

Hebrew marks the speaker's gender in the letters (אני הולך / אני הולכת, הייתי עייף / הייתי עייפה). Text meant
to be read aloud by a male or female voice must already be spelled for that speaker before it reaches TTS.
Example use: a Hebrew podcast script with a male and a female character, where each character's lines come
from this model with `m` or `f`. Forms whose gender shows only in the vowels (אני רוצה: rotse/rotsa) are
spelled the same for both genders. They need a gender-aware nikud model, which is the planned stage 2,
[Hebrew-Gendered-Nikud](https://github.com/Hebrish-ASR/Hebrew-Gendered-Nikud).

This is a merged LoRA fine-tune of [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it),
which is Apache-2.0 licensed.

## Usage

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Hebrish/Hebrew-Gendered-Sentence-Creator"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

SYSTEM = ("Write exactly n different natural Hebrew sentences, unpointed, 6 to 18 words each, that each use the term "
          "once, in the meaning the context describes; the term may take a prefix and may be inflected to agree with "
          "the \"I\". The sentences are in the first person, and every first-person form agrees with speaker_gender "
          "(m = the \"I\" is a man, f = a woman); other people keep their own gender. Reply with a JSON array of strings.")
user = json.dumps({"term": "מילואים", "context": "IDF reserve duty", "n": 5, "speaker_gender": "f"}, ensure_ascii=False)
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=600, do_sample=True, temperature=0.8, top_p=0.95)
print(json.loads(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)))
```

- Use exactly this system prompt; the model was trained with it.
- `speaker_gender` is required and must be `m` or `f`.
- The reply is a JSON array of `n` strings.

## Training

- **Data:** [`Hebrish/Hebrew-Gendered-Sentences`](https://huggingface.co/datasets/Hebrish/Hebrew-Gendered-Sentences).
  - Claude Sonnet 5.5 generated 934 everyday Hebrew terms with context lines. For each term and gender it then
    wrote 10 sentences, natively in that gender.
  - Filters: a format gate (length, term present, no nikud) and a first-person gender check with
    [`dicta-il/dictabert-joint`](https://huggingface.co/dicta-il/dictabert-joint).
  - Result: 17,725 sentences kept, 6,884 training examples over 834 terms. 100 terms are held out.
- **LoRA:** rank 32, alpha 32, dropout 0.05, on the language model's attention and MLP projections only.
  lr 2e-4, cosine schedule, 2 epochs, effective batch 16 (4 x 4), 861 steps. Loss on the reply only.
- **Hardware:** one NVIDIA L40S on Modal, 1,935 s.

## Evaluation

100 held-out terms x {m, f}, n=10, temperature 0.8. A "disagree" is a first-person form whose letters show
the wrong gender, as judged by the dictabert-joint analyser with rules. The analyser has some false alarms,
for example טוב or נוח used about something other than the speaker.

| | Gemma 4 E4B, prompt only | v1 | v2 | **v3 (this)** |
|---|---|---|---|---|
| unparseable JSON (of 200) | 6 | 0 | 1 | 3 |
| gate pass | 1,855/1,940 | 1,956/2,000 | 1,921/1,990 | 1,902/1,970 |
| female "I": agree / disagree | 510 / 19 | 668 / 41 | 680 / 26 | 658 / 22 |
| male "I": agree / disagree | 488 / 2 | 773 / 3 | 740 / 0 | 762 / 2 |
| sentences whose letters show the speaker's gender | 1,019 | 1,485 | 1,446 | 1,440 |

Compared with prompting the base model, the fine-tune:
- breaks the JSON format less often;
- writes about 40% more sentences in which the speaker's gender is actually visible;
- writes more natural, varied Hebrew. This is a qualitative reading by the developer's assistant; no human
  evaluation has been done yet.

Its gender accuracy for a female speaker is not yet better than prompting.

## Limitations

- **Masculine adjective terms with a female speaker.** If the term is given in its masculine dictionary form
  and describes the speaker (מדוכא, אסיר תודה, שוכר, צמחוני), the model often copies it unchanged for
  `f` ("אני מדוכא"). That accounts for 17 of v3's 22 female errors. The cause is too few training examples
  where the term inflects (134).
  - **Workaround:** pass the term already inflected for the speaker (`"term": "מדוכאה"`).
  - **Fix planned for v4:** targeted data for about 300 self-description terms.
- **No guarantee.** A fine-tune is still probabilistic. To be sure, check each sentence's first-person forms
  with a morphological analyser and regenerate the rejects; the analyser code is in the GitHub repo
  (`scripts/analyse_modal.py`).
- **Content.** Occasional factual or lexical slips, for example "האטיג", or an etrog on Yom Kippur.
- **Vowel-only gender** (רוצה, קונה, יפה) is out of scope; see stage 2.

## Licence

The weights are Apache-2.0, following the Gemma 4 base. The training code is MIT
([GitHub](https://github.com/Hebrish-ASR/Hebrew-Gendered-Sentence-Creator)).
