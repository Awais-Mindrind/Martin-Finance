import os
import shutil

RAW_DIR = "/workspace/data/raw_pdfs"

# Allows user to change archive target without modifying code
ARCHIVE_DIR = os.environ.get(
    "PDF_ARCHIVE_PATH",
    "/workspace/data/archive"
)

def main():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    files = os.listdir(RAW_DIR)
    if not files:
        print("No PDFs found in raw_pdfs. Nothing to archive.")
        return

    for f in files:
        if f.lower().endswith(".pdf") or f.lower().endswith(".txt"):
            src = os.path.join(RAW_DIR, f)
            dst = os.path.join(ARCHIVE_DIR, f)
            shutil.move(src, dst)
            print(f"Archived: {f}")

    print("===================================================")
    print("All used PDF/TXT files were moved to:")
    print(ARCHIVE_DIR)
    print("===================================================")

if __name__ == "__main__":
    main()
