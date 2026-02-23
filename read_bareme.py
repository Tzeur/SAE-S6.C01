import sys
from pypdf import PdfReader

if len(sys.argv) < 2:
    print("Usage: python read_bareme.py <pdf_path>")
    sys.exit(1)

pdf_path = sys.argv[1]
reader = PdfReader(pdf_path)
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
print(text)
