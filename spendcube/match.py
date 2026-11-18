"""Count each purchase once.

1. Cross-system: a secondary-system record (e.g. AP-tool invoice line) that is
   the same purchase as a primary record (ERP receipt) is excluded as DUPLICATE
   and linked to it (so it inherits the primary's supplier/item/category).
   Evidence, strongest first:
     a. same PO number + same quantity + same net amount
     b. same PO + net amount equals the sum of 2-3 primary lines of that PO
        (one invoice covering several receipts)
     c. same quantity + same net amount + near-identical line text
        (PO missing or mistyped)
2. Within a system: the same record exported twice (configured key columns)
   counts once; the later copies are DUPLICATE.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .clean import norm_text


def _tol(v, pct, absol):
    return np.maximum(absol, np.abs(v) * pct)


def cross_match(df, cfg):
    mc = cfg.get("matching")
    link = pd.Series("", index=df.index, dtype=object)
    how = pd.Series("", index=df.index, dtype=object)
    if not mc:
        return link, how
    P = df[df["source"] == mc["primary"]]
    S = df[df["source"] == mc["secondary"]]
    pct = mc.get("amount_tolerance_pct", 0.005)
    absol = mc.get("amount_tolerance_abs", 0.05)
    tthr = mc.get("text_threshold", 85)

    p = pd.DataFrame({"pidx": P.index, "po": P["po"].values, "q": P["quantity"].round(4).values,
                      "amt": P["net_local"].values, "txt": P["description"].map(norm_text).values})
    s = pd.DataFrame({"sidx": S.index, "po": S["po"].values, "q": S["quantity"].round(4).values,
                      "amt": S["net_local"].values, "txt": S["description"].map(norm_text).values})

    # a. PO + qty + amount (+ text as tie-break)
    a = s.dropna(subset=["po"]).merge(p.dropna(subset=["po"]), on=["po", "q"], suffixes=("", "_p"))
    a = a[(a["amt"] - a["amt_p"]).abs() <= _tol(a["amt_p"], pct, absol)]
    if len(a):
        a["ts"] = [fuzz.ratio(x, y) for x, y in zip(a["txt"], a["txt_p"])]
        a = a.sort_values(["sidx", "ts"], ascending=[True, False]).drop_duplicates("sidx")
        link[a["sidx"].values] = df.loc[a["pidx"].values, "record_id"].values
        how[a["sidx"].values] = "PO_QTY_AMOUNT"

    # b. same PO, invoice amount = sum of 2..3 receipts on that PO
    rest = s[(link[s["sidx"]].values == "") & s["po"].notna()]
    pg = {k: g for k, g in p.groupby("po")}
    for r in rest.itertuples():
        g = pg.get(r.po)
        if g is None or len(g) < 2 or len(g) > 40:
            continue
        vals = g["amt"].values
        found = None
        for k in (2, 3):
            for comb in combinations(range(len(vals)), k):
                if abs(vals[list(comb)].sum() - r.amt) <= max(absol, abs(r.amt) * pct):
                    found = comb
                    break
            if found:
                break
        if found:
            link[r.sidx] = df.loc[g["pidx"].values[found[0]], "record_id"]
            how[r.sidx] = "PO_SUM_OF_RECEIPTS"

    # c. qty + amount + text (PO missing / not matching)
    rest = s[link[s["sidx"]].values == ""].copy()
    rest["ar"] = rest["amt"].round(0)
    pp = p.copy()
    pp["ar"] = pp["amt"].round(0)
    frames = []
    for d in (0, -1, 1):  # neighbouring rounding buckets
        x = pp.copy()
        x["ar"] = x["ar"] + d
        frames.append(rest.merge(x, on=["q", "ar"], suffixes=("", "_p")))
    c = pd.concat(frames, ignore_index=True)
    c = c[(c["amt"] - c["amt_p"]).abs() <= _tol(c["amt_p"], pct, absol)]
    if len(c):
        c["ts"] = [fuzz.ratio(x, y) for x, y in zip(c["txt"], c["txt_p"])]
        c = c[c["ts"] >= tthr].sort_values(["sidx", "ts"], ascending=[True, False]).drop_duplicates("sidx")
        link[c["sidx"].values] = df.loc[c["pidx"].values, "record_id"].values
        how[c["sidx"].values] = "QTY_AMOUNT_TEXT"
    return link, how


def duplicate_groups(df, cfg):
    """Within one source, rows identical on the configured keys are one record
    exported more than once. Returns a group label per row (-1 = no group).
    duplicate_keys: list of canonical/'src.' columns, or 'ALL_SOURCE_COLUMNS'
    (every original column except the id)."""
    grp = pd.Series(-1, index=df.index, dtype=np.int64)
    nxt = 0
    for sc in cfg["sources"]:
        keys = sc.get("duplicate_keys")
        if not keys:
            continue
        m = df["source"] == sc["name"]
        if keys == "ALL_SOURCE_COLUMNS":
            idc = "src." + sc.get("id_column", "")
            keys = [c for c in df.columns if c.startswith("src.") and c != idc and df.loc[m, c].notna().any()]
        sub = df.loc[m, keys].copy()
        for k in keys:
            if sub[k].dtype == float:
                sub[k] = sub[k].round(2)
            else:
                sub[k] = sub[k].fillna("").astype(str)
        g = sub.groupby(keys, dropna=False, sort=False).ngroup()
        size = g.map(g.value_counts())
        g = g.where(size > 1, -1)
        grp[g.index] = np.where(g >= 0, g + nxt, -1)
        nxt += int(g.max()) + 1 if (g >= 0).any() else 0
    return grp
