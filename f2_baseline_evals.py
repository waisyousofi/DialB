import os, json, argparse
import pandas as pd
import sacrebleu


def clean_output(text):
    marker = "### Response:"
    if marker in text:
        text = text.split(marker, 1)[-1]
    return text.strip()


def load_refs(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def main(args):
    with open(args.output_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    outputs = data["outputs"]

    refs = load_refs(args.ref_txt_path)

    start_id = meta.get("start_id", 1)
    end_id = meta.get("end_id", len(outputs))
    refs = refs[start_id - 1:end_id]

    modes = meta.get("modes")
    if not modes:
        modes = [k for k in outputs[0].keys() if k != "sentence_id"]

    rows = []

    for mode in modes:
        hyps = [clean_output(x[mode]) for x in outputs]

        if len(hyps) != len(refs):
            raise ValueError(
                f"Length mismatch for {mode}: "
                f"{len(hyps)} hypotheses vs {len(refs)} references"
            )

        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
        chrfpp = sacrebleu.corpus_chrf(
            hyps,
            [refs],
            char_order=6,
            word_order=2,
            beta=2
        )

        rows.append({
            "model_name": meta["model_name"],
            "corpus_name": meta["corpus_name"],
            "direction": meta["direction"],
            "decoding_strategy": mode,
            "chrf": chrfpp,
            "bleu": round(bleu, 4),
            "num_sentences": len(hyps),
            "output_json_path": args.output_json_path,
            "ref_txt_path": args.ref_txt_path,
        })

    df = pd.DataFrame(rows)
    
    output_path = args.output_json_path.replace(".json",".csv")
    
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_json_path", required=True)
    parser.add_argument("--ref_txt_path", required=True)
    args = parser.parse_args()
    main(args)