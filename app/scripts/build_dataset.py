import json
import os
from pypdf import PdfReader

PRETEST = "/workspace/data/processed/pdf_pretest.json"
RAW_DIR = "/workspace/data/raw_pdfs"
OUT_JSONL = "/workspace/data/processed/train.jsonl"

def chunk_text(text, max_chars=1000):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    out, cur = [], ""
    for p in paragraphs:
        if len(cur) + len(p) + 1 <= max_chars:
            cur += ("\n" + p) if cur else p
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out

def main():
    with open(PRETEST) as f:
        meta = json.load(f)

    good = [m["file"] for m in meta if m["recommended"]]

    if not good:
        print("No recommended PDFs found. Check the pretest JSON.")
        return

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)

    count = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as out:
        for fname in good:
            path = os.path.join(RAW_DIR, fname)
            reader = PdfReader(path)
            text = "\n".join(
                [p.extract_text() or "" for p in reader.pages]
            )
            for chunk in chunk_text(text):
                out.write(json.dumps({"text": chunk}, ensure_ascii=False) + "\n")
                count += 1

    print(f"Dataset ready: {OUT_JSONL} ({count} chunks)")

if __name__ == "__main__":
    main()
