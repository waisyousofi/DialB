#!/usr/bin/env python3
"""
Run one Farsi->Dari inference with manual beam search and store, at every decoding
step, each beam's top-k next-token candidates, their scores, and the selected
beam expansions.

Example:
CUDA_VISIBLE_DEVICES=0 python beam_trace_inference.py \
  --model_name google/gemma-3-27b-it \
  --sentence "تقریباً چه مدت است که این علایم ادامه دارد؟" \
  --sd Farsi --td Dari --parent_lang Persian \
  --num_beams 4 --top_k 10 --max_new_tokens 128 \
  --out_json beam_trace_sentence1.json
"""

import argparse
import json
import math
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "You are an expert in {parent_lang} dialect translation.\n"
    "Translate the following sentence from {sd} to {td}, preserving meaning while adapting vocabulary, morphology, syntax, and dialect-specific usage to natural {td}.\n"
    "Input: {sd_input_sentence}\n"
    "Return only the translated sentence without any explanation for it.\n"
    "### Response:"
)


def build_prompt(sentence: str, sd: str, td: str, parent_lang: str) -> str:
    return PROMPT_TEMPLATE.format(
        parent_lang=parent_lang,
        sd=sd,
        td=td,
        sd_input_sentence=sentence,
    )


def clean_token_text(tokenizer, token_id: int) -> str:
    """Decode one token in a visualization-friendly way."""
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def extract_response(decoded_text: str) -> str:
    marker = "### Response:"
    if marker in decoded_text:
        return decoded_text.split(marker, 1)[-1].strip()
    return decoded_text.strip()


@torch.no_grad()
def trace_manual_beam_search(
    model,
    tokenizer,
    prompt: str,
    num_beams: int = 4,
    top_k: int = 10,
    max_new_tokens: int = 128,
    length_penalty: float = 1.0,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Manual beam search for one prompt.

    At each step:
      1. For every active beam, compute top_k next-token log-probabilities.
      2. Combine previous beam score + next-token score.
      3. Pick the best num_beams global expansions.
      4. Store all top_k candidates and the selected expansions.

    This is intentionally transparent rather than optimized.
    """
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    prompt_ids = encoded["input_ids"][0]
    prompt_len = int(prompt_ids.shape[0])

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id

    beams: List[Dict[str, Any]] = [
        {
            "beam_id": 0,
            "tokens": prompt_ids.clone(),
            "score": 0.0,
            "finished": False,
        }
    ]

    trace_steps: List[Dict[str, Any]] = []

    for step in range(max_new_tokens):
        all_candidates_for_display = []
        global_expansions = []

        for beam_pos, beam in enumerate(beams):
            if beam["finished"]:
                # Keep finished beams alive so the beam count stays stable.
                candidate = {
                    "rank": 1,
                    "token_id": eos_token_id,
                    "token": clean_token_text(tokenizer, eos_token_id),
                    "token_logprob": 0.0,
                    "cumulative_score": float(beam["score"]),
                    "prob": 1.0,
                    "is_eos": True,
                }
                all_candidates_for_display.append(
                    {
                        "beam_pos": beam_pos,
                        "beam_id": beam["beam_id"],
                        "prefix_text": tokenizer.decode(beam["tokens"][prompt_len:], skip_special_tokens=True),
                        "beam_score_before": float(beam["score"]),
                        "finished": True,
                        "top_candidates": [candidate],
                    }
                )
                global_expansions.append(
                    {
                        "from_beam_pos": beam_pos,
                        "from_beam_id": beam["beam_id"],
                        "token_id": eos_token_id,
                        "token": clean_token_text(tokenizer, eos_token_id),
                        "token_logprob": 0.0,
                        "new_score": float(beam["score"]),
                        "new_tokens": beam["tokens"],
                        "finished": True,
                    }
                )
                continue

            input_ids = beam["tokens"].unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids, device=device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

            values, indices = torch.topk(log_probs, k=top_k)

            top_candidates = []
            for rank, (tok_score, tok_id) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
                cumulative = float(beam["score"] + tok_score)
                top_candidates.append(
                    {
                        "rank": rank,
                        "token_id": int(tok_id),
                        "token": clean_token_text(tokenizer, int(tok_id)),
                        "token_logprob": float(tok_score),
                        "cumulative_score": cumulative,
                        "prob": float(math.exp(tok_score)) if tok_score > -100 else 0.0,
                        "is_eos": int(tok_id) == eos_token_id,
                    }
                )
                global_expansions.append(
                    {
                        "from_beam_pos": beam_pos,
                        "from_beam_id": beam["beam_id"],
                        "token_id": int(tok_id),
                        "token": clean_token_text(tokenizer, int(tok_id)),
                        "token_logprob": float(tok_score),
                        "new_score": cumulative,
                        "new_tokens": torch.cat([beam["tokens"], torch.tensor([tok_id], device=beam["tokens"].device)]),
                        "finished": int(tok_id) == eos_token_id,
                    }
                )

            all_candidates_for_display.append(
                {
                    "beam_pos": beam_pos,
                    "beam_id": beam["beam_id"],
                    "prefix_text": tokenizer.decode(beam["tokens"][prompt_len:], skip_special_tokens=True),
                    "beam_score_before": float(beam["score"]),
                    "finished": False,
                    "top_candidates": top_candidates,
                }
            )

        # Pick next beams globally.
        global_expansions.sort(key=lambda x: x["new_score"], reverse=True)
        selected = global_expansions[:num_beams]

        new_beams = []
        selected_for_json = []
        for new_pos, exp in enumerate(selected):
            new_beam_id = int(step * num_beams + new_pos + 1)
            new_beams.append(
                {
                    "beam_id": new_beam_id,
                    "tokens": exp["new_tokens"].detach().cpu(),
                    "score": float(exp["new_score"]),
                    "finished": bool(exp["finished"]),
                }
            )
            selected_for_json.append(
                {
                    "new_beam_pos": new_pos,
                    "new_beam_id": new_beam_id,
                    "from_beam_pos": int(exp["from_beam_pos"]),
                    "from_beam_id": int(exp["from_beam_id"]),
                    "token_id": int(exp["token_id"]),
                    "token": exp["token"],
                    "token_logprob": float(exp["token_logprob"]),
                    "new_score": float(exp["new_score"]),
                    "finished": bool(exp["finished"]),
                    "new_text": tokenizer.decode(exp["new_tokens"][prompt_len:], skip_special_tokens=True),
                }
            )

        trace_steps.append(
            {
                "step": step + 1,
                "beams_before_selection": all_candidates_for_display,
                "selected_expansions": selected_for_json,
            }
        )

        beams = new_beams
        if all(b["finished"] for b in beams):
            break

    # Rank final beams with optional length penalty.
    final_beams = []
    for pos, beam in enumerate(beams):
        gen_len = max(1, int(beam["tokens"].shape[0] - prompt_len))
        normalized_score = beam["score"] / (gen_len ** length_penalty)
        decoded_full = tokenizer.decode(beam["tokens"], skip_special_tokens=True)
        decoded_new = tokenizer.decode(beam["tokens"][prompt_len:], skip_special_tokens=True)
        final_beams.append(
            {
                "beam_pos": pos,
                "beam_id": int(beam["beam_id"]),
                "raw_score": float(beam["score"]),
                "length_normalized_score": float(normalized_score),
                "generated_text": decoded_new.strip(),
                "response_only": extract_response(decoded_full),
                "finished": bool(beam["finished"]),
            }
        )

    final_beams.sort(key=lambda x: x["length_normalized_score"], reverse=True)

    return {
        "prompt": prompt,
        "prompt_len_tokens": prompt_len,
        "num_beams": num_beams,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "trace_steps": trace_steps,
        "final_beams": final_beams,
        "best_output": final_beams[0]["response_only"] if final_beams else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--sentence", type=str, required=True)
    parser.add_argument("--sd", type=str, default="Farsi")
    parser.add_argument("--td", type=str, default="Dari")
    parser.add_argument("--parent_lang", type=str, default="Persian")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--out_json", type=str, default="beam_trace.json")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype_map[args.torch_dtype],
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    device = model.device
    prompt = build_prompt(args.sentence, args.sd, args.td, args.parent_lang)
    print("prompt",prompt)
    
    trace = trace_manual_beam_search(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_beams=args.num_beams,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        length_penalty=args.length_penalty,
        device=str(device),
    )

    trace["metadata"] = {
        "model_name": args.model_name,
        "source_dialect": args.sd,
        "target_dialect": args.td,
        "parent_lang": args.parent_lang,
        "source_sentence": args.sentence,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    print(f"Saved trace to: {args.out_json}")
    print("Best output:")
    print(trace["best_output"])


if __name__ == "__main__":
    main()
