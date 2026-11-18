"""Decide what counts as spend. Every excluded record gets one reason code.

Order of precedence (first hit wins): CANCELLED, NOT_APPROVED, NON_PROCUREMENT,
DUPLICATE (set later by the matcher / self-dedup).
"""
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")


def _rx(p):
    return re.compile(p, re.IGNORECASE)


def apply_status(df, cfg):
    reason = pd.Series("", index=df.index, dtype=object)
    rules = cfg.get("status_rules", {})
    for sc in cfg["sources"]:
        st = sc.get("status") or {}
        if not st.get("column"):
            continue
        m = df["source"] == sc["name"]
        s = df.loc[m, "status"].fillna("")
        inc = _rx(st.get("include_pattern", rules.get("include_pattern", r"^$")))
        can = _rx(st.get("cancelled_pattern", rules.get("cancelled_pattern", r"$^")))
        nap = _rx(st.get("not_approved_pattern", rules.get("not_approved_pattern", r"$^")))
        blank_ok = st.get("blank_counts", rules.get("blank_counts", True))
        # unknown status -> NOT_APPROVED (only approved or paid counts)
        r = pd.Series("NOT_APPROVED", index=s.index, dtype=object)
        r[s.map(lambda x: bool(inc.search(x)) and not nap.search(x))] = ""
        r[s.map(lambda x: bool(can.search(x)))] = "CANCELLED"
        if blank_ok:
            r[s == ""] = ""
        reason[m] = r
    return reason


def _rule_mask(df, rule):
    col = rule.get("field") or ("src." + rule["column"])
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col].fillna("").astype(str)
    m = s.str.contains(rule["pattern"], flags=re.IGNORECASE, regex=True)
    if rule.get("unless"):
        m &= ~s.str.contains(rule["unless"], flags=re.IGNORECASE, regex=True)
    if rule.get("sources"):
        m &= df["source"].isin(rule["sources"])
    return m


def apply_rules(df, cfg, reason):
    """Config-driven rules: [{field|column, pattern, unless?, reason, sources?}]."""
    hits = pd.Series("", index=df.index, dtype=object)
    for rule in cfg.get("exclusion_rules", []) + cfg.get("default_exclusion_rules", []):
        m = _rule_mask(df, rule) & (reason == "") & (hits == "")
        hits[m] = rule.get("reason", "NON_PROCUREMENT")
    reason = reason.copy()
    reason[(reason == "") & (hits != "")] = hits[(reason == "") & (hits != "")]
    return reason
