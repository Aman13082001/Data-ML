"""Parsing helpers: numbers, VAT rates, dates, FX, text normalisation."""
import re
import unicodedata

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- numbers
def parse_number(s):
    """Parse amounts written as '1,234.56', '1.234,56', '1 234,56', '(12.00)', '12-'."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    t = str(s).strip().replace(" ", " ")
    if t == "" or t.lower() in {"nan", "none", "null", "-"}:
        return np.nan
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1]
    if t.endswith("-"):
        neg, t = True, t[:-1]
    # currency symbols/codes may lead or trail ('R 1,234.00', '1234 ZAR'); letters
    # inside the value mean it is not a number ('V500009', 'PO-123')
    t = re.sub(r"^([A-Za-z]{1,3}\s+|[^\w\s\-+.,(]+)", "", t)
    t = re.sub(r"(\s+[A-Za-z]{1,3}|%|[^\w\s.,)]+)$", "", t)
    t = t.replace(" ", "")
    if re.search(r"[^\d,.\-+]", t) and not re.fullmatch(r"-?\d+(\.\d+)?[eE][-+]?\d+", t):
        return np.nan
    if "," in t and "." in t:
        # whichever separator comes last is the decimal mark
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        # '1,234' (thousands) vs '12,5' (decimal)
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+", t):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
    try:
        v = float(t)
    except ValueError:
        return np.nan
    return -v if neg else v


def parse_numbers(series):
    return series.map(parse_number).astype(float)


def parse_rate(series, default=None):
    """VAT rates written as '15', '15%', '0.15' -> 0.15. Blank -> default."""
    v = series.astype(str).str.replace("%", "", regex=False).map(parse_number)
    v = np.where(v > 1, v / 100.0, v)
    v = pd.Series(v, index=series.index, dtype=float)
    if default is not None:
        v = v.fillna(default)
    return v


# ---------------------------------------------------------------- dates
DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d-%b-%y", "%d-%b-%Y", "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%d/%m/%y",
    "%Y-%m-%d %H:%M:%S", "%m/%d/%Y",
]


def parse_dates(series, formats=None, dayfirst=True, valid_from=None, valid_to=None):
    """Try each format in turn; return (dates, flag) where flag is
    '' ok, 'MISSING', 'UNPARSEABLE' or 'OUT_OF_RANGE'."""
    fmts = list(formats or DATE_FORMATS)
    if not dayfirst and "%m/%d/%Y" in fmts:
        fmts.remove("%m/%d/%Y")
        fmts.insert(0, "%m/%d/%Y")
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    todo = s.notna() & (s != "")
    for f in fmts:
        if not todo.any():
            break
        p = pd.to_datetime(s[todo], format=f, errors="coerce")
        hit = p.notna()
        out.loc[p.index[hit]] = p[hit]
        todo.loc[p.index[hit]] = False
    flag = pd.Series("", index=series.index)
    flag[s.isna() | (s == "")] = "MISSING"
    flag[todo] = "UNPARSEABLE"
    if valid_from:
        bad = out < pd.Timestamp(valid_from)
        flag[bad] = "OUT_OF_RANGE"
    if valid_to:
        bad = out > pd.Timestamp(valid_to)
        flag[bad] = "OUT_OF_RANGE"
    return out, flag


# ---------------------------------------------------------------- FX
def fx_lookup(fx, currencies, months):
    """fx: DataFrame[currency, month(Period M), rate]. Returns a rate per row,
    using the exact month or else the nearest month for that currency."""
    rates = np.full(len(currencies), np.nan)
    by_cur = {c: g.sort_values("month") for c, g in fx.groupby("currency")}
    for i, (c, m) in enumerate(zip(currencies, months)):
        g = by_cur.get(c)
        if g is None or len(g) == 0:
            continue
        if m is None or pd.isna(m):
            rates[i] = g["rate"].median()
            continue
        d = (g["month"].map(lambda p: p.ordinal) - m.ordinal).abs()
        rates[i] = g["rate"].iloc[int(np.argmin(d.values))]
    return rates


# ---------------------------------------------------------------- text
def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


OCR_MAP = {"0": "O", "1": "I", "5": "S", "8": "B"}
_OCR_MID = re.compile(r"(?<=[A-Z])[0158](?=[A-Z])")
_OCR_START = re.compile(r"(?<![A-Z0-9])[0158](?=[A-Z]{3,})")


def fix_ocr(t):
    """Map look-alike digits to letters only where they sit inside a word:
    'R0D' -> 'ROD', 'B1T' -> 'BIT', '5HAFTS' -> 'SHAFTS'. '4KW', '22KW' untouched."""
    t = _OCR_MID.sub(lambda m: OCR_MAP[m.group(0)], t)
    return _OCR_START.sub(lambda m: OCR_MAP[m.group(0)], t)


def norm_text(s, ocr=True):
    """Upper-case, strip accents, unify decimal commas, fix OCR digits, collapse spaces."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    t = strip_accents(str(s)).upper()
    t = t.replace("&QUOT;", '"').replace("&AMP;", "&")
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)          # 2,4M -> 2.4M
    if ocr:
        t = fix_ocr(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compact(s):
    """Letters and digits only: 'OAK ENT MINING' -> 'OAKENTMINING'."""
    return re.sub(r"[^A-Z0-9]", "", s)
