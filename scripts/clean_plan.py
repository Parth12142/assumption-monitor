import re
import sys
from collections import Counter

import pandas as pd

# ---------- 1. Read pages ----------
def load_pages(path):
    import pymupdf
    doc = pymupdf.open(path)
    return [page.get_text() for page in doc]

# ---------- 2. Find header/footer lines ----------
def find_repeated_lines(pages, min_share=0.6):
    counts = Counter()
    for text in pages:
        for line in set(l.strip() for l in text.split("\n")):
            if line:
                counts[line] += 1
    limit = min_share * len(pages)
    return {line for line, c in counts.items() if c >= limit}

PAGE_NUM = re.compile(r"^Page\s+\d+$", re.IGNORECASE)
HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")

# ---------- 3. Clean one page into paragraphs ----------
def page_to_paragraphs(text, repeated):
    paragraphs, current = [], []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line in repeated or PAGE_NUM.match(line):
            continue
        if HEADING.match(line) and not line.endswith("."):
            if current:
                paragraphs.append(("TEXT", " ".join(current)))
                current = []
            paragraphs.append(("HEADING", line))
            continue
        if len(line) <= 25 and not line.endswith((".", "!", "?")):
            continue  # table cells and tiny fragments
        current.append(line)
    if current:
        paragraphs.append(("TEXT", " ".join(current)))
    return paragraphs

# ---------- 4. Split into sentences ----------
ABBREVIATIONS = ["Rs.", "Pvt.", "Ltd.", "No.", "Dr.", "Mr.", "Mrs.", "vs.", "approx."]
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

def split_sentences(paragraph):
    for abbr in ABBREVIATIONS:
        paragraph = paragraph.replace(abbr, abbr.replace(".", "<DOT>"))
    parts = SPLIT.split(paragraph)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]

# ---------- 5. Regex features ----------
PERCENT = re.compile(r"\d+(?:\.\d+)?\s?%")
MONEY = re.compile(r"(?:Rs\.|USD|\$)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:crore|lakh|million|billion))?")
CUE = re.compile(r"\b(?:assum\w*|expect\w*|project\w*|believe\w*|estimat\w*|forecast\w*|anticipat\w*)\b", re.IGNORECASE)

def build_table(pages):
    repeated = find_repeated_lines(pages)
    rows, section = [], "(start)"
    for page_no, text in enumerate(pages, start=1):
        for kind, content in page_to_paragraphs(text, repeated):
            if kind == "HEADING":
                section = content
                continue
            for sent in split_sentences(content):
                rows.append({"page": page_no, "section": section, "sentence": sent})
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="sentence").reset_index(drop=True)
    df["n_words"] = df["sentence"].str.split().str.len()
    df = df[df["n_words"] >= 5].reset_index(drop=True)
    df["percents"] = df["sentence"].apply(PERCENT.findall)
    df["money"] = df["sentence"].apply(MONEY.findall)
    df["has_cue"] = df["sentence"].str.contains(CUE)
    df.insert(0, "sentence_id", range(1, len(df) + 1))
    return df

if __name__ == "__main__":
    pages = load_pages(sys.argv[1])
    df = build_table(pages)
    out = sys.argv[2] if len(sys.argv) > 2 else "data/clean/sentences.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} sentences saved to {out}")
    print(df[["sentence_id", "page", "has_cue", "sentence"]].to_string(max_colwidth=80))
