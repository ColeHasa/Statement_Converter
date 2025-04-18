import streamlit as st
import pdfplumber, fitz, openai, tempfile, os, base64, csv, re
import pandas as pd
from dateutil import parser
from io import StringIO
from dotenv import load_dotenv

# ───────────────────  CONFIG  ────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

COMMON_PROMPT = (
    "Extract all financial transactions from the statement, including charges, purchases, deposits, payments, credits, and refunds. "
    "Each row must include exactly three fields: Date, Description, and Amount. "
    "The Amount should appear exactly as shown on the statement, including whether it is positive or negative, but with commas removed (e.g., 1000.00 instead of 1,000.00). "
    "Return only raw CSV with a header row: Date,Description,Amount — no formatting, no backticks, and no extra commentary."
)

# ──────────────────  GPT HELPERS  ─────────────────
def gpt_from_image(img_bytes: bytes) -> str:
    data_url = f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": COMMON_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()

def gpt_from_text(text: str) -> str:
    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": COMMON_PROMPT + "\n\n" + text}],
        max_tokens=7500
    )
    return resp.choices[0].message.content.strip()

# ─────────────────  PDF HELPERS  ──────────────────
def text_layer(path: str) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages)

def image_ocr(path: str) -> str:
    doc, parts = fitz.open(path), []
    for i, page in enumerate(doc):
        st.info(f"OCR page {i + 1}")
        parts.append(gpt_from_image(page.get_pixmap(dpi=150).tobytes("png")))
    return "\n".join(parts)

# ────────────────  CSV SANITISER  ─────────────────
FENCE_RE = re.compile(r"""^[\s`'"“”‘’]{0,5}(?:`{3,}|'{3,}|"{3,})?\s*(?:csv)?\s*$""", re.I)
COMMENT_RE = re.compile(r"^\s*(here\s+is|note:)", re.I)

def clean_csv_text(txt: str) -> str:
    return "\n".join(
        l for l in txt.splitlines()
        if l.strip() and not (FENCE_RE.match(l) or COMMENT_RE.match(l))
    )

# ────────────────  NORMALISERS  ───────────────────
PAREN_RE = re.compile(r"^\(\s*\$?([0-9,.]+)\s*\)$")

def normalise_amount(raw: str) -> str:
    txt = raw.replace(",", "").replace("$", "").strip()
    if (m := PAREN_RE.match(txt)):
        return f"-{m.group(1)}"
    if txt.endswith("-"):
        return f"-{txt[:-1].strip()}"
    return txt

def normalise_date(raw: str, default_year: int) -> str:
    txt = raw.strip()
    if not txt:
        return ""
    txt = re.sub(r"[‑–—−]", "/", txt)
    try:
        dt = parser.parse(txt, dayfirst=False, fuzzy=True, default=pd.Timestamp(f"{default_year}-01-01"))
        return dt.strftime("%m/%d/%Y")
    except Exception:
        return ""

# ────────────────  FINAL CSV PARSER  ────────────────
HEADER_PREFIXES = ("date", "description", "amount", "here", "note", "csv")

def robust_parse(csv_text: str, default_year: int) -> pd.DataFrame:
    csv_text = clean_csv_text(csv_text)
    reader = csv.reader(StringIO(csv_text), delimiter=",", quotechar='"')
    rows = []

    for row in reader:
        if not row or row[0].strip().lower().startswith(HEADER_PREFIXES):
            continue
        if len(row) < 3:
            continue

        date_raw = row[0].strip()
        amount_raw = row[-1].strip()
        description = ",".join(row[1:-1]).strip()

        date_fixed = normalise_date(date_raw, default_year)
        if not date_fixed:
            continue

        amount_fixed = normalise_amount(amount_raw)
        rows.append([date_fixed, description, amount_fixed])

    return pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

# ────────────────  STREAMLIT APP  ────────────────
def file_key(upl) -> str:
    return f"{upl.name}_{upl.size}_{upl.type}"

def main():
    st.title("Bank PDF → QuickBooks CSV  (cached, GPT‑4o)")

    upl = st.file_uploader("Upload bank‑statement PDF", type="pdf")
    if not upl:
        st.info("👆 Upload a PDF to begin.")
        return

    key = file_key(upl)

    if st.session_state.get("cached_key") != key:
        with st.spinner("Processing PDF…"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(upl.read())
                pdf_path = tmp.name

            txt = text_layer(pdf_path)

            if txt.strip():
                year_source = txt
                raw_csv = gpt_from_text(txt)
            else:
                raw_csv = image_ocr(pdf_path)
                year_source = raw_csv

            year_match = re.search(r"\b(20\d{2})\b", year_source)
            default_year = int(year_match.group(1)) if year_match else pd.Timestamp.now().year

            os.remove(pdf_path)

            df = robust_parse(raw_csv, default_year)

            st.session_state.update(
                cached_key=key,
                cached_csv=raw_csv,
                cached_df=df,
                cached_year=default_year
            )
    else:
        raw_csv = st.session_state.cached_csv
        df = st.session_state.cached_df
        default_year = st.session_state.cached_year

    st.subheader("Raw CSV (editable)")
    edited = st.text_area("Review / edit CSV below", value=clean_csv_text(raw_csv), height=300)

    df_display = robust_parse(edited, default_year)
    if df_display.empty:
        st.info("⚠️ No transaction rows detected after cleaning. "
                "Check for unusual date formats in the raw CSV above.")
        return

    st.subheader(f"Preview  ({len(df_display)} rows)")
    st.dataframe(df_display, use_container_width=True)

    st.download_button(
        "Download CSV for QuickBooks",
        data=df_display.to_csv(index=False).encode("utf-8"),
        file_name="StatementConverterCSV.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
