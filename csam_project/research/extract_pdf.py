from pypdf import PdfReader
import sys

try:
    reader = PdfReader("c:/Users/lamaq/OneDrive/Desktop/CSAM project/csam_project/research/Sem 5 Memory Optimisation (1).pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    output_path = "c:/Users/lamaq/OneDrive/Desktop/CSAM project/csam_project/research/extracted_pdf_content.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Successfully extracted {len(text)} characters to {output_path}")
    print("Preview of start:")
    print(text[:500])
except Exception as e:
    print(f"Error: {e}")
