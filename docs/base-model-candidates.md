# Base-model candidates

Researched 2026-11-18 from HF model cards and API metadata, and from the Hebrew LLM Leaderboard. [C] means
confirmed on the page; [I] means inferred. Nothing has been run on the task yet. Phase 0 of
[`training-plan.md`](training-plan.md) makes the choice.

What the task needs, in order: natural Hebrew generation; first-person gender agreement in the letters (the
"I" of the sentence, usually the speaker); JSON discipline; small enough to fine-tune for a few dollars and to
run cheaply; a licence that allows a public release under Hebrish.

## Sources

- Base-model board (Dicta + Mafat): https://huggingface.co/spaces/hebrew-llm-leaderboard/leaderboard.
  "Avg" below is its average over SNLI, HeQ QA, sentiment, Winograd, translation BLEU and Israeli trivia.
  These are comprehension tasks, not free generation or gender agreement.
- Chat board: https://huggingface.co/spaces/hebrew-llm-leaderboard/chat-leaderboard (dataset
  `hebrew-llm-leaderboard/chat-results`). It has no models of 12B or under.
- Tok/word: tokens per Hebrew word, measured with each model's `tokenizer.json` on a 72-word,
  6-sentence first-person Hebrew sample. The sample is small, so the numbers are indicative only.

## Shortlist for the phase-0 probe

| # | Model (instruct variant for the probe) | Params | Licence | Hebrew | Leaderboard Avg (base) | Tok/word | Why it is on the list |
|---|---|---|---|---|---|---|---|
| 1 | `dicta-il/DictaLM-3.0-1.7B-Instruct` | 1.7B | apache-2.0 [C] | Hebrew continued pretraining on Qwen3-1.7B-Base [C] | 51.5 | ~1.8 [I, Qwen3 tokenizer; its tokenizer.json returned 404] | **Recommended main track.** The cheapest Hebrew-trained model. Scores at the level of 4B general models. Plain `Qwen3ForCausalLM` [I], so the English sibling's training script ports almost unchanged. |
| 2 | `google/gemma-4-E4B-it` | 8.0B incl. embeddings ("E4B") | apache-2.0 [C, leaderboard] | Claimed multilingual (140+ languages); Hebrew not named [C] | 60.7 | 2.46 | The strongest general model at small effective size. The English plan already listed it for round 2. Multimodal wrapper, so it needs more setup. |
| 3 | `dicta-il/DictaLM-3.0-Nemotron-12B-Instruct` | 12.3B, hybrid Mamba-2 | NVIDIA Open Model Licence [C] | Hebrew + English continued pretraining [C] | 66.5 (best ≤12B) | 2.47 | Quality ceiling within budget. Risks: needs `trust_remote_code` [C]; PEFT/Unsloth LoRA support for Nemotron-H must be checked; the licence is not Apache. |
| 4 | `dicta-il/dictalm2.0-instruct` | 7.25B, Mistral-7B based | apache-2.0 [C] | Hebrew-extended vocabulary, 190B tokens of continued pretraining, 50% Hebrew [C] | 62.6 (instruct 61.5) | 2.32 | A mature, plain Mistral architecture with strong Hebrew. Fallback if the 1.7B is not fluent enough. |
| 5 | `Qwen/Qwen3-4B` | 4.0B | apache-2.0 [C] | Claimed (119 languages); Hebrew not named | 53.1 | 1.81 | Control: the same family as #1, but without the Hebrew pretraining. Shows what Dicta's continued pretraining adds. |
| 6 | `microsoft/Phi-4-mini-instruct` | 3.8B | MIT [C] | **Hebrew listed explicitly** [C] | no row | 1.94 | A cheap tokenizer with explicit Hebrew support. Unscored, so the probe decides. |
| 7 | `google/gemma-4-E2B-it` | 5.1B incl. embeddings | apache-2.0 [C] | Claimed | 50.0 | 2.46 | The small Gemma, against #1 at similar cost. |

Optional extra: `google/gemma-3-12b-it` (gated Gemma licence, Avg 64.0). It is a plain transformer, so it is
the safe option if #3's architecture blocks LoRA.

## Ruled out

| Model | Reason |
|---|---|
| `ibm-granite/granite-4.2-3b` / `-8b` (the English sibling's base) | Hebrew is not a listed language [C]. 5.36 tokens/word, 2 to 3 times the others, so training and inference cost more. |
| `meta-llama/Llama-3.2-3B` / `3.1-8B` | Avg 30.2 (3B) and 52.3 (8B), Hebrew not listed. 5.29 tokens/word (3B). Gated licence. |
| `CohereLabs/aya-expanse-8b`, `aya-23-8B` | Hebrew is explicit, but cc-by-nc-4.0 blocks a public Hebrish release for general use. |
| `mistralai/Ministral-3-*-2512` | The card's language list leaves out Hebrew [C]. No leaderboard row. |
| `yam-peleg/Hebrew-Mistral-7B`, `Hebrew-Gemma-11B` | 2024 models, superseded by DictaLM 2.0/3.0 (Avg 51.8 and 50.6). |
| `dicta-il/DictaLM-3.0-24B-*`, `DictaLM-4.0-105B-A10B` | Above the size budget (Avg 72.5 and 70.0). Usable as **distillation writers or judges**, not as the base. |
| `Qwen/Qwen3.5-*` | 2.6 tokens/word, no leaderboard rows ≤9B, and linear attention needs extra training kernels (from the English plan). Revisit only if Qwen3 does well. |
| `ai21labs/AI21-Jamba2-3B` | The card makes no claim for Hebrew. Hybrid architecture. |

## Recommendation

Probe #1 to #7 zero-shot (16 calls each, as in the training plan's phase 0), then pilot LoRA on the best Dicta
model and the best general model. The working bet is **DictaLM-3.0-1.7B-Instruct**. It is Apache-2.0, small
enough for local CPU, and Hebrew-trained, and its architecture is one the English pipeline already handles.
If it is not fluent enough after the pilot, step up to dictalm2.0-instruct (7B, Apache) or Nemotron-12B
(stronger, but the licence and architecture cost more).

Not yet verified: the Gemma-4-12B base repo names; the DictaLM-3 technical report (not read); whether the
Dicta models are on OpenRouter (if not, the probe runs them on Modal vLLM).
