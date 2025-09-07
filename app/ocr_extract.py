import subprocess
from pypdf import PdfReader
import pdfplumber

def has_text_layer(pdf_path: str) -> bool:
    try:
        reader = PdfReader(pdf_path)
        sample_pages = min(3, len(reader.pages))
        extracted = ""
        for i in range(sample_pages):
            extracted += reader.pages[i].extract_text() or ""
        return len(extracted.strip()) > 30
    except Exception:
        return False

def ensure_searchable_pdf(input_pdf: str, output_pdf: str) -> None:
    if has_text_layer(input_pdf):
        with open(input_pdf, "rb") as src, open(output_pdf, "wb") as dst:
            dst.write(src.read())
        return
    cmd = [
        "ocrmypdf",
        "--quiet",
        "--skip-text",
        "--optimize", "3",
        "--language", "por+eng",
        input_pdf,
        output_pdf
    ]
    subprocess.run(cmd, check=True)

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            t = page.extract_text() or ""
            text += t + "\n"
    except Exception:
        text = ""

    if len(text.strip()) < 50:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                parts = []
                for page in pdf.pages:
                    parts.append(page.extract_text() or "")
                text = "\n".join(parts)
        except Exception:
            pass

    return text or ""
