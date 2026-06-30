import os
import csv
import statistics


CSV_PATH = "/home/waisullah/root/storage/dialectal_corpora/Arabic/multidialectal_travel_corpus.csv?download=1"

OUTPUT_DIR = "/home/waisullah/root/storage/unsw_expt_outputs/dialectal_mt"
OUTPUT_FILE = "arabic_multidialectal_travel_corpus_stats.txt"


def clean_sentence(text):
    text = str(text).strip()

    prefixes = [
        "- Saudi:",
        "- Egyptian:",
        "- Iraqi:",
        "- Levantine:",
        "- Moroccan:",
    ]

    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1).strip()

    return text


def word_count(sentence):
    return len(sentence.split())


def get_stats(sentences):
    lengths = [word_count(s) for s in sentences if s.strip()]

    return {
        "num_sentences": len(lengths),
        "total_words": sum(lengths),
        "avg_words": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "min_words": min(lengths) if lengths else 0,
        "max_words": max(lengths) if lengths else 0,
    }


def main():
    dialect_sentences = {}

    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for col in reader.fieldnames:
            dialect_sentences[col] = []

        for row in reader:
            for col in reader.fieldnames:
                sent = clean_sentence(row[col])
                if sent:
                    dialect_sentences[col].append(sent)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("ARABIC MULTIDIALECTAL TRAVEL CORPUS STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"CSV path: {CSV_PATH}\n\n")

        for dialect, sentences in dialect_sentences.items():
            stats = get_stats(sentences)

            f.write(f"Dialect: {dialect}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sentences    : {stats['num_sentences']}\n")
            f.write(f"Total words  : {stats['total_words']}\n")
            f.write(f"Avg words    : {stats['avg_words']}\n")
            f.write(f"Min words    : {stats['min_words']}\n")
            f.write(f"Max words    : {stats['max_words']}\n")

    print("Saved stats report to:")
    print(out_path)


if __name__ == "__main__":
    main()