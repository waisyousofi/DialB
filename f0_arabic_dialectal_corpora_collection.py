import os
import csv
import urllib.request


URL = "https://zenodo.org/records/17776617/files/multidialectal_travel_corpus.csv?download=1"

OUTPUT_DIR = "/home/waisullah/root/storage/dialectal_corpora/Arabic"

CSV_FILENAME = "multidialectal_travel_corpus.csv"
CSV_PATH = os.path.join(OUTPUT_DIR, CSV_FILENAME)


DIALECT_COLUMNS = {
    "MSA": "Modern Standard Arabic",
    "Saudi": "Saudi Dialect",
    "Egyptian": "Egyptian Dialect",
    "Iraqi": "Iraqi Dialect",
    "Levantine": "Levantine Dialect",
    "Moroccan": "Moroccan Dialect",
}


PREFIXES_TO_REMOVE = [
    "- Saudi:",
    "- Egyptian:",
    "- Iraqi:",
    "- Levantine:",
    "- Moroccan:",
]


def download_csv(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"CSV already exists, skipping download: {output_path}")
        return

    print(f"Downloading:\n{url}")
    print(f"Saving to:\n{output_path}")

    urllib.request.urlretrieve(url, output_path)

    print("Download finished.")


def clean_sentence(text):
    text = str(text).strip()

    for prefix in PREFIXES_TO_REMOVE:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1).strip()

    return text


def export_dialect_txt_files(csv_path, output_dir):
    dialect_sentences = {dialect: [] for dialect in DIALECT_COLUMNS}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        print("CSV columns found:")
        print(reader.fieldnames)

        for row in reader:
            for dialect_name, column_name in DIALECT_COLUMNS.items():
                sent = clean_sentence(row.get(column_name, ""))

                if sent:
                    dialect_sentences[dialect_name].append(sent)

    for dialect_name, sentences in dialect_sentences.items():
        out_path = os.path.join(output_dir, f"{dialect_name}.txt")

        with open(out_path, "w", encoding="utf-8") as f:
            for sent in sentences:
                f.write(sent + "\n")

        print(f"Saved {dialect_name}: {len(sentences)} sentences")
        print(out_path)


def main():
    download_csv(URL, CSV_PATH)
    export_dialect_txt_files(CSV_PATH, OUTPUT_DIR)

    print("\nDone.")
    print(f"CSV saved at: {CSV_PATH}")
    print(f"Dialect txt files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()