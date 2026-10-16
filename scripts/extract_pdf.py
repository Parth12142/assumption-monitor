import sys
import fitz  # PyMuPDF

path = sys.argv[1]
doc = fitz.open(path)
print("pages:", len(doc))

for i, page in enumerate(doc, start=1):
    text = page.get_text()
    print(f"--- page {i}: {len(text)} characters ---")
    print(text[:400])
