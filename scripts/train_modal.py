"""LoRA SFT of Gemma 4 (text only) on data/train.jsonl, then generation on data/eval.jsonl.

    MODAL_PROFILE=danielrosehill with-secrets modal run scripts/train_modal.py \
        --base google/gemma-4-E4B-it --run gemma-4-e4b-v1 [--epochs 2 --lr 2e-4 --rank 32]
    smoke test: --run gemma-smoke --limit-train 200 --limit-eval 4 --epochs 1 --no-push

Ported from Term-Context-Sentence-Generator/scripts/train_modal.py (2026-11-18). Differences: Gemma 4 is a
multimodal checkpoint (Gemma4ForConditionalGeneration), so LoRA targets only the language model's linear
layers (regex below), and transformers is 5.x. Loss on the assistant turn only. Merged weights are pushed
privately to danielrosehill/hebrew-gendered-sentence-creator-<run>. Eval generations for both genders go to
data/runs/<run>/eval-outputs.jsonl.
"""
import json, os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
HF_OWNER = "danielrosehill"
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.1", "transformers==5.23.0", "accelerate", "peft==0.21.3", "huggingface_hub", "pillow",
                 "torchvision")
    .add_local_file(ROOT / "data/train.jsonl", "/data/train.jsonl")
    .add_local_file(ROOT / "data/eval.jsonl", "/data/eval.jsonl")
    .add_local_file(ROOT / "scripts/system_prompt.py", "/root/system_prompt.py")
)
app = modal.App("hebrew-gendered-sentence-creator-train", image=image)
hf = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]} if modal.is_local() else {})
from system_prompt import SYSTEM  # noqa: E402  (scripts/system_prompt.py, shared with build_dataset.py)
MICRO = 4
LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def prompt(tok, user):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@app.function(gpu="L40S", timeout=4 * 3600, secrets=[hf], env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
def train(base: str, run: str, epochs: float, lr: float, rank: int, eval_n: int, limit_train: int, limit_eval: int, push: bool):
    import math, random, time
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).cuda()
    print("loaded", type(model).__name__, flush=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=LORA_TARGETS))
    model.print_trainable_parameters()
    # Gemma 4 closes a turn with "<turn|>" (id 106, one of the config's eos ids); Gemma 3 used "<end_of_turn>"
    end = next((t for t in ("<turn|>", "<end_of_turn>") if t in tok.get_vocab()), tok.eos_token)

    def encode(r):
        p = tok(prompt(tok, r["messages"][1]["content"]), add_special_tokens=False).input_ids
        t = tok(r["messages"][2]["content"] + end, add_special_tokens=False).input_ids
        return p + t, [-100] * len(p) + t

    lines = open("/data/train.jsonl").readlines()
    data = [encode(json.loads(l)) for l in (lines[:limit_train] if limit_train else lines)]
    print(len(data), "examples; max len", max(len(x) for x, _ in data), "end token", repr(end), flush=True)
    print(prompt(tok, json.loads(lines[0])["messages"][1]["content"]), flush=True)
    bs, log = 16, []
    steps = math.ceil(len(data) / bs * epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
    model.train()
    random.seed(0)
    step, t0 = 0, time.time()
    while step < steps:
        random.shuffle(data)
        chunks = [sorted(data[i:i + bs * 50], key=lambda x: len(x[0])) for i in range(0, len(data), bs * 50)]
        batches = [c[k:k + bs] for c in chunks for k in range(0, len(c), bs)]
        random.shuffle(batches)
        for b in batches:
            if step >= steps: break
            # micro-batches of 4: Gemma 4's 262k-token vocabulary makes batch-16 logits too big for 48 GB
            for k in range(0, len(b), MICRO):
                mb = b[k:k + MICRO]
                L = max(len(x) for x, _ in mb)
                ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x, _ in mb]).cuda()
                lab = torch.tensor([y + [-100] * (L - len(y)) for _, y in mb]).cuda()
                att = (torch.arange(L)[None, :] < torch.tensor([len(x) for x, _ in mb])[:, None]).cuda()
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss * len(mb) / len(b)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0 or step == steps:
                log.append({"step": step, "loss": round(loss.item() * len(b) / len(mb), 4), "s": round(time.time() - t0)})
                print(f"step {step}/{steps} loss {log[-1]['loss']:.4f} {time.time() - t0:.0f}s", flush=True)
    train_s = time.time() - t0

    model = model.merge_and_unload()
    model.eval()
    tok.padding_side = "left"
    ev = [json.loads(l) for l in open("/data/eval.jsonl")]
    ev = ev[:limit_eval] if limit_eval else ev
    jobs = [(r, g) for r in ev for g in ("m", "f")]
    outs, t1 = [], time.time()
    for k in range(0, len(jobs), 8):
        b = jobs[k:k + 8]
        users = [json.dumps({"term": r["term"], "context": r["context"], "n": eval_n, "speaker_gender": g},
                            ensure_ascii=False) for r, g in b]
        enc = tok([prompt(tok, u) for u in users], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=80 * eval_n + 40, do_sample=True, temperature=0.8, top_p=0.95,
                               pad_token_id=tok.pad_token_id)
        for (r, g), x in zip(b, o):
            outs.append({"term": r["term"], "context": r["context"], "gender": g, "n": eval_n,
                         "raw": tok.decode(x[enc.input_ids.shape[1]:], skip_special_tokens=True)})
    gen_s = time.time() - t1

    repo = f"{HF_OWNER}/hebrew-gendered-sentence-creator-{run}" if push else None
    if push:
        model.push_to_hub(repo, private=True)
        tok.push_to_hub(repo, private=True)
        print("pushed", repo, flush=True)
    return {"run": run, "base": base, "hf_repo": repo, "examples": len(data), "steps": steps, "epochs": epochs,
            "lr": lr, "rank": rank, "train_s": round(train_s), "eval_gen_s": round(gen_s), "loss_log": log}, outs


@app.local_entrypoint()
def main(base: str = "google/gemma-4-E4B-it", run: str = "gemma-4-e4b-v1", epochs: float = 2.0,
         lr: float = 2e-4, rank: int = 32, eval_n: int = 10, limit_train: int = 0, limit_eval: int = 0,
         push: bool = True):
    info, outs = train.remote(base, run, epochs, lr, rank, eval_n, limit_train, limit_eval, push)
    d = ROOT / "data" / "runs" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "train.json").write_text(json.dumps(info, indent=1))
    with open(d / "eval-outputs.jsonl", "w") as f:
        for o in outs: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "loss_log"}))
