"""Read every source file described in the config into one canonical frame.

Canonical columns (one row per input record):
  record_id, source, role, amount_raw, net_local, currency, fx_rate, spend,
  date, date_flag, supplier_name, supplier_id, description, item_code,
  quantity, po, po_item, invoice, status, text_extra, plus 'src.<col>' copies
  of every original column so rules in the config can reference them.
"""
import re

import numpy as np
import pandas as pd

from .clean import fx_lookup, parse_dates, parse_numbers, parse_rate
from .config import data_path, read_table, resolve_file


def _col(df, name):
    if name is None:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    if name not in df.columns:
        raise KeyError(f"column '{name}' not found; available: {list(df.columns)}")
    return df[name]


def _join_text(df, cols):
    if not cols:
        return pd.Series([""] * len(df), index=df.index)
    if isinstance(cols, str):
        cols = [cols]
    parts = [df[c].fillna("").astype(str).str.strip() for c in cols if c in df.columns]
    out = parts[0]
    for p in parts[1:]:
        out = out.str.cat(p, sep=" ")
    return out.str.replace(r"\s+", " ", regex=True).str.strip()


def _extract(series, pattern):
    if pattern is None:
        return series.astype("string").str.strip()
    return series.astype("string").str.extract(pattern, expand=False)


def load_fx(cfg):
    fxc = cfg.get("fx")
    if not fxc or not fxc.get("file"):
        return None
    fx = read_table(resolve_file(cfg, fxc["file"], [fxc.get("currency_column", "currency"), fxc.get("rate_column", "rate")]))
    fx = pd.DataFrame({
        "currency": fx[fxc.get("currency_column", "currency")].str.strip().str.upper(),
        "month": pd.to_datetime(fx[fxc.get("month_column", "month")], errors="coerce").dt.to_period("M"),
        "rate": parse_numbers(fx[fxc.get("rate_column", "rate")]),
    }).dropna()
    return fx


def load_source(cfg, sc, fx):
    req = [c for c in [sc.get("id_column"), (sc.get("amount") or {}).get("column")] if c]
    # new extracts are never taken from the build folder: fallback only for files that are not records
    df = read_table(resolve_file(cfg, sc["file"], req, fallback=False), sep=sc.get("sep"),
                    encoding=sc.get("encoding"), sheet=sc.get("sheet", 0))
    df.columns = [str(c).strip() for c in df.columns]
    n = len(df)
    out = pd.DataFrame(index=df.index)
    idc = sc.get("id_column")
    raw_id = df[idc].astype(str) if idc else pd.Series([f"{i+1}" for i in range(n)], index=df.index)
    out["raw_id"] = raw_id
    out["record_id"] = sc.get("id_prefix", "") + raw_id
    out["source"] = sc["name"]
    out["role"] = sc.get("role", "primary")

    # ---- amounts
    am = sc["amount"]
    amt = parse_numbers(_col(df, am["column"])) if am.get("column") else pd.Series(np.nan, index=df.index)
    if am.get("quantity_times_unit_price"):
        # no line amount: quantity x unit price
        amt = parse_numbers(_col(df, am["quantity_column"])) * parse_numbers(_col(df, am["unit_price_column"]))
    out["amount_raw"] = amt
    if am.get("gross") and am.get("tax_amount_column"):
        # gross amount with the tax given as an amount: net = gross - tax
        tax = parse_numbers(_col(df, am["tax_amount_column"])).fillna(0.0)
        out["vat_rate"] = np.where(amt - tax != 0, tax / (amt - tax), 0.0)
        out["net_local"] = amt - tax
    elif am.get("gross"):
        rate = parse_rate(_col(df, am.get("vat_rate_column")), am.get("default_vat_rate", 0.0)) \
            if am.get("vat_rate_column") else pd.Series(am.get("default_vat_rate", 0.0), index=df.index)
        out["vat_rate"] = rate
        out["net_local"] = amt / (1.0 + rate)
    else:
        out["vat_rate"] = 0.0
        out["net_local"] = amt
    if am.get("sign_column"):
        # e.g. a debit/credit indicator column; values listed in negative_values flip the sign
        neg = df[am["sign_column"]].astype(str).str.strip().str.upper().isin(
            [v.upper() for v in am.get("negative_values", [])])
        out.loc[neg, "net_local"] = -out.loc[neg, "net_local"].abs()

    # ---- dates (first column that parses wins; others are fallbacks)
    dc = sc.get("date", {})
    dcols = dc.get("columns", [])
    date = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    flag = pd.Series("MISSING", index=df.index)
    for i, c in enumerate(dcols):
        d, f = parse_dates(_col(df, c), dc.get("formats"), dc.get("dayfirst", True),
                           dc.get("valid_from"), dc.get("valid_to"))
        if i == 0:
            flag = f.copy()
        take = date.isna() & (f == "")
        date[take] = d[take]
        if i > 0:
            flag[take & (flag != "")] = "FALLBACK_" + c
    out["date"] = date
    out["date_flag"] = flag

    # ---- currency -> reporting currency
    rep = cfg["reporting_currency"].upper()
    cc = sc.get("currency", {})
    if cc.get("column"):
        cur = df[cc["column"]].fillna(cc.get("default", rep)).astype(str).str.strip().str.upper()
    else:
        cur = pd.Series(cc.get("fixed", rep), index=df.index)
    out["currency"] = cur
    rate = pd.Series(np.nan, index=df.index)
    rate[cur == rep] = 1.0
    if cc.get("rate_column"):
        r = parse_numbers(df[cc["rate_column"]])
        r[(r <= 0)] = np.nan
        use = rate.isna() & r.notna()
        rate[use] = r[use]
    out["fx_flag"] = ""
    if fx is not None:
        # month for the table lookup: configured column, fallback to the parsed date
        mcol = cc.get("month_from")
        if mcol:
            md, mf = parse_dates(_col(df, mcol), dc.get("formats"), dc.get("dayfirst", True),
                                 dc.get("valid_from"), dc.get("valid_to"))
            md = md.where(mf == "", out["date"])
        else:
            md = out["date"]
        months = md.dt.to_period("M")
        foreign = cur != rep
        table = pd.Series(np.nan, index=df.index)
        table[foreign] = fx_lookup(fx, cur[foreign].tolist(), months[foreign].tolist())
        # a given rate far from the table rate for that month is a data error
        # (e.g. another currency's rate on the line): use the table instead
        f = cc.get("rate_sanity_factor")
        if f:
            ratio = rate / table
            bad = foreign & rate.notna() & table.notna() & ((ratio > f) | (ratio < 1.0 / f))
            rate[bad] = table[bad]
            out.loc[bad, "fx_flag"] = "RATE_REPLACED_BY_TABLE"
        need = rate.isna() & table.notna()
        rate[need] = table[need]
        out.loc[need, "fx_flag"] = "FX_TABLE"
    out.loc[rate.isna(), "fx_flag"] = "NO_FX_RATE"
    out["fx_rate"] = rate
    if cc.get("rate_direction", "units_per_reporting") == "units_per_reporting":
        out["spend"] = out["net_local"] / rate
    else:  # reporting per unit
        out["spend"] = out["net_local"] * rate

    # ---- supplier / item / text
    sup = sc.get("supplier", {})
    out["supplier_name"] = _col(df, sup.get("name_column")).fillna("").astype(str) if sup.get("name_column") else ""
    out["supplier_id"] = _col(df, sup.get("id_column")).fillna("").astype(str) if sup.get("id_column") else ""
    out["description"] = _join_text(df, sc.get("description"))
    # text that identifies the item (defaults to the description)
    out["item_text"] = _join_text(df, sc["item_description"]) if sc.get("item_description") else out["description"]
    out["item_code"] = _col(df, sc.get("item_code")).fillna("").astype(str) if sc.get("item_code") else ""
    out["quantity"] = parse_numbers(_col(df, sc["quantity"])) if sc.get("quantity") else np.nan
    out["text_extra"] = _join_text(df, sc.get("classifier_text"))
    out["cat_extra"] = _join_text(df, sc.get("classifier_tokens")) if sc.get("classifier_tokens") else ""

    po = sc.get("po", {})
    if po.get("column"):
        out["po"] = _extract(df[po["column"]], po.get("pattern"))
        if po.get("item_column"):
            out["po_item"] = parse_numbers(df[po["item_column"]])
        elif po.get("item_pattern"):
            out["po_item"] = parse_numbers(_extract(df[po["column"]], po["item_pattern"]))
        else:
            out["po_item"] = np.nan
    else:
        out["po"], out["po_item"] = None, np.nan
    inv = sc.get("invoice", {})
    if inv.get("column"):
        out["invoice"] = df[inv["column"]].astype(str)
        out["invoice_digits"] = df[inv["column"]].astype(str).str.replace(r"\D", "", regex=True).str.lstrip("0")
    else:
        out["invoice"], out["invoice_digits"] = "", ""
    st = sc.get("status", {})
    out["status"] = df[st["column"]].fillna("").astype(str).str.strip().str.lower() if st.get("column") else ""

    for c in df.columns:
        out["src." + c] = df[c]
    return out


def load_all(cfg):
    fx = load_fx(cfg)
    frames = [load_source(cfg, sc, fx) for sc in cfg["sources"]]
    return pd.concat(frames, ignore_index=True, sort=False)
