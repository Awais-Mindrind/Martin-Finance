import os, json
from pypdf import PdfReader

RAW = "/workspace/data/raw_pdfs"
OUT = "/workspace/data/processed/pdf_pretest.json"

def pretest(path):
    reader = PdfReader(path)
    pages = reader.pages
    text = "\n".join(p.extract_text() or "" for p in pages)

    num_pages = len(pages)
    num_chars = len(text)
    alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))

    score = 0
    if num_pages >= 2: score += 20
    if num_chars > 2000: score += 30
    if alpha_ratio > 0.6: score += 30
    if num_chars < 100000: score += 20
    return dict(
        num_pages=num_pages,
        num_chars=num_chars,
        alpha_ratio=alpha_ratio,
        score=score,
        recommended=score >= 60
    )

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    for f in os.listdir(RAW):
        if f.lower().endswith(".pdf"):
            info = pretest(os.path.join(RAW,f))
            info["file"] = f
            results.append(info)
    with open(OUT,"w") as f:
        json.dump(results,f,indent=2)

if __name__=="__main__":
    main()
