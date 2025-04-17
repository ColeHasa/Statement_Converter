import streamlit as st
import pdfplumber
import pandas as pd
from io import BytesIO
import tempfile
import os
import base64
import openai
import re
import fitz  # PyMuPDF

# Set your OpenAI API key
openai.api_key = ""  # 🔐 Replace with your key

def gpt_transaction_extraction(image_bytes):
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{encoded}"

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "From this bank statement, extract only the transaction rows. Each transaction must have exactly 3 fields: Date, Description, and Amount. Format the response as raw CSV with a header row: Date,Description,Amount. Do not include balances, totals, summaries, or any additional explanations. If a line is missing any of the three fields, skip it. Also do not add additional headers for each page converted, only a header at the very beginning. it should read as a header at the top and then rtansactions listed one after another from all pages being converted."},
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT Vision failed: {e}")
        return ""

def gpt_transaction_extraction_from_text(text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": f"Extract only the transactions from the following bank statement text. Format the result as raw CSV with three columns: Date, Description, Amount. Do not include any explanation, just the CSV.\n\n{text}"
                }
            ],
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"GPT Text Extraction failed: {e}")
        return ""

def extract_text_from_text_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_images_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    results = []
    for i, page in enumerate(doc):
        st.info(f"Rendering page {i+1} for GPT OCR...")
        pix = page.get_pixmap(dpi=150)
        image_bytes = pix.tobytes("png")
        response = gpt_transaction_extraction(image_bytes)
        results.append(response)
    return "\n".join(results)

from io import StringIO

def parse_csv_output(csv_text):
    try:
        return pd.read_csv(StringIO(csv_text), skip_blank_lines=True)
    except Exception as e:
        st.warning(f"Failed to parse CSV output: {e}")
        return pd.DataFrame()

def main():
    st.title("Bank PDF to QuickBooks CSV Converter (GPT with CSV Output)")

    uploaded_file = st.file_uploader("Upload a bank statement PDF", type="pdf")

    if uploaded_file:
        raw_bytes = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(raw_bytes)
            tmp_file_path = tmp_file.name

        try:
            text = extract_text_from_text_pdf(tmp_file_path)
            if not text.strip():
                raise ValueError("Empty text from pdfplumber, using image-based OCR...")
            st.success("Extracted text. Sending to GPT for CSV formatting...")
            gpt_output = gpt_transaction_extraction_from_text(text)
        except:
            st.warning("Text extraction failed. Using image-based OCR with GPT...")
            gpt_output = extract_images_from_pdf(tmp_file_path)

        os.remove(tmp_file_path)

        st.subheader("GPT Extracted Transactions (CSV Format)")
        st.text_area("Review or edit the extracted CSV", value=gpt_output, height=300)

        df = parse_csv_output(gpt_output)
        if not df.empty:
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV for QuickBooks", csv, "qbo_upload.csv", "text/csv")
        else:
            st.error("Could not parse GPT output. Please review the extracted content manually.")

if __name__ == "__main__":
    main()
