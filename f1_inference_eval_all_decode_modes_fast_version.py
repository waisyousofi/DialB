import os
import json
import argparse
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================================================
# PROMPT
# =========================================================

PROMPT_TEMPLATE = (
    "### Instruction:\n"
    "You are an expert in {parent_lang} dialect translation.\n"
    "Translate the following sentence from {sd} to {td}, preserving meaning while adapting vocabulary, morphology, syntax, and dialect-specific usage to natural {td}.\n"
    "Input: {sd_input_sentence}\n"
    "Return only the translated sentence without any explanation for it.\n"
    "### Response:"
)

def build_prompt(sentence, sd, td, parent_lang):
    return PROMPT_TEMPLATE.format(
        parent_lang=parent_lang,
        sd=sd,
        td=td,
        sd_input_sentence=sentence
    )

# =========================================================
# DATA
# =========================================================

def load_txt_sentences(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    # remove empty lines
    lines = [x for x in lines if x]

    return lines


def slice_sentences(sentences, start_id, end_id):
    """
    User-friendly indexing:
    start_id=1, end_id=300 means first 300 sentences.

    Internally Python uses 0-based indexing.
    """
    start_idx = max(start_id - 1, 0)

    if end_id is None or end_id <= 0:
        end_idx = len(sentences)
    else:
        end_idx = min(end_id, len(sentences))

    return sentences[start_idx:end_idx], start_idx, end_idx


# =========================================================
# OUTPUT CLEANING
# =========================================================

def extract_response(decoded_text):
    """
    Remove the prompt part if the model returns prompt + completion.
    """
    marker = "### Response:"
    if marker in decoded_text:
        return decoded_text.split(marker, 1)[-1].strip()

    return decoded_text.strip()


def safe_model_name(model_name):
    return model_name.replace("/", "_")


# =========================================================
# GENERATION MODES
# =========================================================

def generation_config_for_mode(args, mode):
    """
    Return kwargs for model.generate().
    """

    if mode == "greedy":
        return {
            "do_sample": False,
            "num_beams": 1,
        }

    if mode == "beam":
        return {
            "do_sample": False,
            "num_beams": args.num_beams        }

    if mode == "dbs":
        if args.num_beams % args.num_beam_groups != 0:
            raise ValueError(
                f"For DBS, num_beams must be divisible by num_beam_groups. "
                f"Got num_beams={args.num_beams}, "
                f"num_beam_groups={args.num_beam_groups}"
            )

        if args.num_beam_groups <= 1:
            raise ValueError("For DBS, num_beam_groups must be > 1.")

        return {
            "do_sample": False,
            "num_beams": args.num_beams,
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty        }

    raise ValueError(f"Unknown decoding mode: {mode}")


# =========================================================
# MAIN
# =========================================================

def main(args):

    print("=" * 80)
    print("Dialectal MT decoding baseline")
    print("=" * 80)

    print("Model:", args.model_name)
    print("Source dialect:", args.sd)
    print("Target dialect:", args.td)
    print("Input txt:", args.sd_txt_path)
    print("Corpus:", args.corpus_name)
    print("Output main dir:", args.output_main_dir)

    if torch.cuda.is_available():
        print("CUDA devices:", torch.cuda.device_count())
        print("GPU 0:", torch.cuda.get_device_name(0))
    else:
        print("CUDA not available. Running on CPU.")

    # -----------------------------
    # Load data
    # -----------------------------

    all_sentences = load_txt_sentences(args.sd_txt_path)

    chunk_sentences, start_idx, end_idx = slice_sentences(
        all_sentences,
        start_id=args.start_id,
        end_id=len(all_sentences)
    )
    print(f"Total input sentences: {len(all_sentences)}")
    print(f"Running chunk: {start_idx + 1} to {end_idx}")
    print(f"Chunk size: {len(chunk_sentences)}")

    # -----------------------------
    # Load model
    # -----------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    ).eval()

    # -----------------------------
    # Prepare outputs
    # -----------------------------

    results = []

    modes = args.modes.split(",")

    valid_modes = {"greedy", "beam"}
    for mode in modes:
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Use only: greedy,beam,dbs")

    # -----------------------------
    # Inference
    # -----------------------------

    for batch_start in tqdm(range(0, len(chunk_sentences), args.batch_size)):
        batch_sents = chunk_sentences[batch_start:batch_start + args.batch_size]

        # prompts = [
        #     build_prompt(sentence, args.sd, args.td)
        #     for sentence in batch_sents
        # ]

        prompts = [
            build_prompt(sentence, args.sd, args.td, args.parent_lang)
            for sentence in batch_sents
        ]
        # print("Parent language:", args.parent_lang)
        # print("Start ID:", args.start_id)        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length
        ).to(model.device)

        batch_records = []

        for local_j, src_sentence in enumerate(batch_sents):
            global_idx = start_idx + batch_start + local_j
            
            print("global_idx",global_idx)
            batch_records.append({
                "sentence_id": global_idx + 1,
                # "source_dialect": args.sd,
                # "target_dialect": args.td,
                # "source_sentence": src_sentence,
            })

        for mode in modes:
            gen_kwargs = generation_config_for_mode(args, mode)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **gen_kwargs
                )

            decoded_batch = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True
            )

            for j, decoded in enumerate(decoded_batch):
                # clean_output = extract_response(decoded)
                batch_records[j][mode] = decoded

                # if args.print_outputs:
                #     print("\n" + "-" * 80)
                #     print(f"ID: {batch_records[j]['sentence_id']}")
                #     print(f"SRC: {batch_records[j]['source_sentence']}")
                #     print(f"[{mode}] {clean_output}")

        results.extend(batch_records)

    # -----------------------------
    # Save
    # -----------------------------

    output_dir = os.path.join(args.output_main_dir, "baseline_outputs")
    os.makedirs(output_dir, exist_ok=True)

    direction = f"{args.sd}2{args.td}"
    model_tag = safe_model_name(args.model_name)

    out_filename = (
        f"{model_tag}-{args.corpus_name}-{direction}.json"
    )

    out_path = os.path.join(output_dir, out_filename)

    metadata = {
        "model_name": args.model_name,
        "corpus_name": args.corpus_name,
        "direction": direction,
        "source_dialect": args.sd,
        "target_dialect": args.td,
        "parent_lang": args.parent_lang,        
        "input_path": args.sd_txt_path,
        "start_id": start_idx + 1,
        "end_id": end_idx,
        "num_sentences": len(results),
        "modes": modes,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "max_input_length": args.max_input_length,
            "batch_size": args.batch_size,
            "num_beams": args.num_beams,
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty,
        }
    }

    final_json = {
        "metadata": metadata,
        "outputs": results
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Saved to:")
    print(out_path)
    print("=" * 80)


# =========================================================
# ARGPARSE
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="google/gemma-3-27b-it")

    parser.add_argument("--sd", type=str, default="Farsi")
    parser.add_argument("--td", type=str, default="Dari")

    parser.add_argument("--sd_txt_path", type=str, default=r"/home/waisullah/root/storage/dialectal_corpora/Persian/test_corpora/CODET/Farsi.pes")
    parser.add_argument("--corpus_name", type=str, default="codet")

    parser.add_argument("--output_main_dir", type=str, default=r'/home/waisullah/root/storage/unsw_expt_outputs/dialectal_mt')

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    parser.add_argument(
        "--modes",
        type=str,
        default="greedy,beam",
        help="Comma-separated modes: greedy,beam"
    )

    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--num_beam_groups", type=int, default=2)
    parser.add_argument("--diversity_penalty", type=float, default=1.0)

    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")

    parser.add_argument("--print_outputs", action="store_true")

    parser.add_argument("--parent_lang", type=str, default="Persian")
    parser.add_argument("--start_id", type=int, default=1)
    
    args = parser.parse_args()

    main(args)
    
    
# CUDA_VISIBLE_DEVICES=0,1 python f1_inference_eval_all_modes.py >>log.f1 2>&1