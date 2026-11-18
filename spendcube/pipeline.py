"""End-to-end: ingest -> exclusions -> count once -> keys -> classify -> output."""
import json
import os
import time

import numpy as np
import pandas as pd

from . import classify as C
from .config import data_path, load_config
from .entities import item_keys, supplier_keys
from .exclusions import apply_rules, apply_status
from .ingest import load_all
from .match import cross_match, duplicate_groups
from .taxonomy import load_taxonomy, path_from_labels


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ stage 1-6
def build_records(cfg):
    df = load_all(cfg)
    log(f"loaded {len(df):,} records from {len(cfg['sources'])} source(s)")

    reason = apply_status(df, cfg)
    reason = apply_rules(df, cfg, reason)
    rconf = pd.Series(np.where(reason != "", 0.95, 0.0), index=df.index)

    # count once: cross-system links, then repeated exports inside one system
    link, how = cross_match(df, cfg)
    grp = duplicate_groups(df, cfg)
    for g, idx in df.index.to_series().groupby(grp):
        if g < 0:
            continue
        idx = list(idx)
        linked = [i for i in idx if link[i]]
        if linked:
            for i in idx:
                if not link[i]:
                    link[i], how[i] = link[linked[0]], "REPEAT_OF_MATCHED"
        else:
            open_ = [i for i in idx if reason[i] == ""]
            for i in open_[1:]:
                link[i], how[i] = df.at[open_[0], "record_id"], "REPEATED_EXPORT"
    dup = (link != "") & (reason == "")
    reason[dup] = "DUPLICATE"
    rconf[dup] = np.where(how[dup].isin(["PO_QTY_AMOUNT", "REPEATED_EXPORT", "REPEAT_OF_MATCHED"]), 0.97, 0.85)
    df["exclusion_reason"] = reason
    df["include"] = (reason == "").astype(int)
    df["link"] = link
    df["match_rule"] = how
    df["rule_conf"] = rconf
    log("exclusions: " + json.dumps(reason.replace("", "INCLUDED").value_counts().to_dict()))
    log("cross-system matches: " + json.dumps(how[how != ""].value_counts().to_dict()))

    # keys; a duplicate inherits the keys of the record it duplicates
    df["supplier_key"], df["supplier_cluster"] = supplier_keys(df, cfg)
    pos = pd.Series(df.index, index=df["record_id"])
    li = df["link"] != ""
    src = pos.reindex(df.loc[li, "link"]).values
    df.loc[li, "supplier_key"] = df.loc[src, "supplier_key"].values
    df.loc[li, "supplier_cluster"] = df.loc[src, "supplier_cluster"].values
    df["item_key"] = item_keys(df, cfg, df["supplier_cluster"])
    df.loc[li, "item_key"] = df.loc[src, "item_key"].values
    log(f"suppliers: {df['supplier_key'].nunique():,} keys, items: {df['item_key'].nunique():,} keys")

    bad = df["spend"].isna() & (df["include"] == 1)
    df["spend_flag"] = np.where(bad, "NO_AMOUNT_OR_FX", "")
    df["spend_out"] = np.where(df["include"] == 1, df["spend"].fillna(0.0), 0.0).round(2)
    return df


# ------------------------------------------------------------------ training data
def load_training(cfg, df=None):
    """Labelled rows for this company: (records, leaf_id, item_key) or None."""
    lc = cfg.get("labels")
    if not lc or not lc.get("file") or cfg.get("_hide_labels"):
        return None
    tdir = lc.get("data_dir")
    if tdir and not os.path.isabs(tdir):
        from .config import REPO
        tdir = os.path.join(REPO, tdir)
    if df is None or (tdir and os.path.normcase(os.path.abspath(tdir)) != os.path.normcase(os.path.abspath(cfg["data_dir"]))):
        tcfg = dict(cfg)
        tcfg["data_dir"] = tdir or cfg["data_dir"]
        log(f"building training records from {tcfg['data_dir']}")
        tr = build_records(tcfg)
    else:
        tr = df
    lab_path = data_path({"data_dir": tdir or cfg["data_dir"]}, lc["file"])
    lab = pd.read_csv(lab_path, dtype=str)
    lab["record_id"] = lab[lc.get("id_column", "record_id")].astype(str)
    if lc.get("include_column"):
        lab = lab[lab[lc["include_column"]].astype(str).str.strip().isin(["1", "1.0", "True", "true"])]
    tax = load_taxonomy({**cfg, "data_dir": tdir or cfg["data_dir"]})
    lab["leaf_id"] = path_from_labels(lab, lc["level_columns"], tax)
    lab = lab.dropna(subset=["leaf_id"])
    tr = tr.merge(lab[["record_id", "leaf_id"]], on="record_id")
    gold = lab.set_index("record_id")
    return tr, gold


def transfer_sources(cfg):
    """Labelled lines of *other* companies, used only to learn generic
    word -> category-word associations for an unseen tree."""
    out = []
    for p in cfg.get("transfer_from", []) or []:
        from .config import REPO
        path = p if os.path.isabs(p) else os.path.join(REPO, p)
        ocfg = load_config(path)
        if not os.path.exists(ocfg["data_dir"]):
            log(f"transfer source {p}: data dir missing, skipped")
            continue
        try:
            got = load_training(ocfg)
        except FileNotFoundError as e:
            log(f"transfer source {p}: {e}, skipped")
            continue
        if got is None:
            continue
        tr, _ = got
        tax = load_taxonomy(ocfg)
        lut = dict(zip(tax["leaf_id"], tax["path"]))
        out.append((C.line_text(tr).tolist(), [lut[l] for l in tr["leaf_id"]]))
        log(f"transfer source {p}: {len(tr):,} labelled lines")
    return out


# ------------------------------------------------------------------ stage 7
def classify_records(df, cfg, tax):
    inc = df["include"] == 1
    sub = df[inc]
    got = load_training(cfg, df)
    mode = "supervised" if got is not None else "zero-shot"
    log(f"classifying {inc.sum():,} included lines ({mode}) into {len(tax)} leaves")
    leaf_ids = tax["leaf_id"].tolist()
    if got is not None:
        tr, _ = got
        tr = tr[tr["leaf_id"].isin(leaf_ids)]
        P, classes = C.classify_supervised(
            sub, C.line_text(tr), C.tokens_col(tr, tr["supplier_cluster"]), tr["leaf_id"].values,
            sub["supplier_cluster"], sub["item_key"].values, train_items=tr["item_key"].values,
            sc=cfg.get("supervised", {}))
        P = pd.DataFrame(P, columns=classes).reindex(columns=leaf_ids, fill_value=0.0).values
    else:
        assoc = None
        srcs = transfer_sources(cfg)
        if srcs:
            assoc = C.learn_transfer(srcs)
            log(f"learned {len(assoc):,} word->category-word associations from other companies")
        P = C.classify_zero_shot(sub, tax, sub["supplier_cluster"], sub["item_key"].values, assoc,
                                 cfg.get("zero_shot", {}))
    best, conf = C.hierarchical_decode(P, tax)
    n_levels = max(tax.attrs.get("n_levels", 3), int(tax["depth"].max()))
    paths = tax["path"].values[best]
    for lvl in range(4):
        df[f"l{lvl+1}"] = ""
        df.loc[inc, f"l{lvl+1}"] = [p[lvl] if lvl < len(p) else "" for p in paths]
    df["confidence"] = df["rule_conf"]
    df.loc[inc, "confidence"] = np.clip(conf, 0, 1)
    df["leaf_id"] = ""
    df.loc[inc, "leaf_id"] = tax["leaf_id"].values[best]
    return df, mode


# ------------------------------------------------------------------ output
OUT_COLS = ["record_id", "include", "exclusion_reason", "supplier_key", "item_key", "spend",
            "l1", "l2", "l3", "l4", "confidence"]


def run(config_path, data_dir=None, out_name=None, out_dir=None, hide_labels=False):
    t0 = time.time()
    cfg = load_config(config_path, data_dir)
    cfg["_hide_labels"] = hide_labels
    log(f"company {cfg.get('company')} | data {cfg['data_dir']}")
    tax = load_taxonomy(cfg)
    df = build_records(cfg)
    df, mode = classify_records(df, cfg, tax)
    out = df.drop(columns=["spend"]).rename(columns={"spend_out": "spend"})[OUT_COLS].copy()
    out["confidence"] = out["confidence"].astype(float).round(3)
    name = out_name or cfg.get("output_name", cfg.get("company", "X"))
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cfg.get('output_prefix', 'out')}__{name}.csv")
    out.to_csv(path, index=False)
    diag_cols = ["record_id", "source", "currency", "fx_rate", "fx_flag", "net_local", "date", "date_flag",
                 "spend_flag", "match_rule", "link", "supplier_name", "description", "leaf_id"]
    os.makedirs(os.path.join(out_dir, "diagnostics"), exist_ok=True)
    df[diag_cols].to_csv(os.path.join(out_dir, "diagnostics", f"diag__{name}.csv"), index=False)
    summary = {
        "records": int(len(out)), "included": int(out["include"].sum()),
        "total_spend": round(float(out["spend"].sum()), 2), "currency": cfg["reporting_currency"],
        "exclusions": out["exclusion_reason"].replace("", "INCLUDED").value_counts().to_dict(),
        "suppliers": int(out.loc[out.include == 1, "supplier_key"].nunique()),
        "items": int(out.loc[out.include == 1, "item_key"].nunique()),
        "classifier": mode, "unparsed_dates": int((df["date_flag"] == "UNPARSEABLE").sum()),
        "out_of_range_dates": int((df["date_flag"] == "OUT_OF_RANGE").sum()),
        "no_fx_rate": int((df["fx_flag"] == "NO_FX_RATE").sum()),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, "diagnostics", f"summary__{name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {path}")
    log(json.dumps(summary))
    return df, out, summary
