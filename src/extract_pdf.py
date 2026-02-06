import sys
try:
    from pypdf import PdfReader
except ImportError:
    print("pypdf not installed")
    sys.exit(1)

reader = PdfReader("Sujet/Sujet_S6.AGED-25-26.n.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
with open("pdf_content.txt", "w", encoding="utf-8") as f:
    f.write(text)
