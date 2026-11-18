"""Measure the pipeline against the labels we have.

  python scripts/evaluate.py A        # supplier/item de-dup, supervised CV, zero-shot (labels hidden)
  python scripts/evaluate.py B        # exclusions vs seed labels, supervised CV, zero-shot
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spendcube import classify as C  # noqa: E402
from spendcube.config import load_config  # noqa: E402
from spendcube.pipeline import build_records, load_training, transfer_sources  # noqa: E402
from spendcube.taxonomy import load_taxonomy  # noqa: E402


def pair_f1(pred, gold):
    """Pairwise precision/recall/F1 of a clustering against gold clusters."""
    d = pd.DataFrame({"p": pd.Series(pred).astype(str).values, "g": pd.Series(gold).astype(str).values})
    c2 = lambda n: (n * (n - 1) / 2).sum()
    tp = c2(d.groupby(["p", "g"]).size())
    pp = c2(d.groupby("p").size())
    gp = c2(d.groupby("g").size())
    pred, gold = d["p"].values, d["g"].values
    p, r = tp / pp if pp else 1, tp / gp if gp else 1
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(2 * p * r / (p + r) if p + r else 0, 4),
                n_pred=len(set(pred)), n_gold=len(set(gold)))


def level_acc(pred_paths, gold_paths, w=None):
    w = np.ones(len(pred_paths)) if w is None else np.abs(np.asarray(w))
    out = {}
    for lvl in range(3):
        hit = np.array([len(p) > lvl and len(g) > lvl and p[:lvl + 1] == g[:lvl + 1]
                        for p, g in zip(pred_paths, gold_paths)])
        out[f"L{lvl+1}"] = round(float(hit.mean()), 4)
        out[f"L{lvl+1}_spend"] = round(float((hit * w).sum() / w.sum()), 4)
    return out


def main(company):
    cfg = load_config(f"configs/{company}.yaml")
    tax = load_taxonomy(cfg)
    lut = dict(zip(tax["leaf_id"], tax["path"]))
    df = build_records(cfg)
    tr, gold = load_training(cfg, df)
    print(f"\n== {company}: {len(tr):,} labelled lines")

    if company == "A":
        g = gold.reindex(tr["record_id"])
        print("supplier dedup (ERP lines):", pair_f1(tr["supplier_key"], g["supplier_key"]))
        has = g["item_key"].notna().values      # the client has golden items only for coded materials
        print("item dedup     (ERP lines with a golden item):", pair_f1(tr["item_key"].values[has], g["item_key"].values[has]))
    if "include_column" in cfg.get("labels", {}):
        lab = pd.read_csv(os.path.join(cfg["data_dir"], cfg["labels"]["file"]), dtype=str)
        m = df.merge(lab, left_on="record_id", right_on=cfg["labels"]["id_column"])
        gi = m["include_y"].astype(str).str.strip() == "1"
        print("include/exclude agreement with seed labels:", round(float((gi == (m["include_x"] == 1)).mean()), 4))
        dis = m[gi != (m["include_x"] == 1)]
        print(dis[["record_id", "exclusion_reason_x", "exclusion_reason_y", "description", "src.gl_account_description", "net_local"]].head(25).to_string())
        print(pd.crosstab(m["exclusion_reason_x"].replace("", "INCL"), m["exclusion_reason_y"].fillna("INCL")))

    # ---- supervised, 3-fold CV (item memory on: new records of a known company)
    t0 = time.time()
    kf = KFold(3, shuffle=True, random_state=0)
    pred = np.empty(len(tr), dtype=object)
    for a, b in kf.split(tr):
        A_, B_ = tr.iloc[a], tr.iloc[b]
        P, classes = C.classify_supervised(B_, C.line_text(A_), C.tokens_col(A_, A_["supplier_cluster"]),
                                           A_["leaf_id"].values, B_["supplier_cluster"], B_["item_key"].values,
                                           train_items=A_["item_key"].values, sc=cfg.get("supervised", {}))
        pred[b] = np.array(classes)[P.argmax(axis=1)]
    print(f"supervised 3-fold CV ({time.time()-t0:.0f}s):",
          level_acc([lut[p] for p in pred], [lut[x] for x in tr["leaf_id"]], tr["spend_out"]))

    # ---- zero-shot: labels hidden, tree only (+ associations from the *other* company)
    t0 = time.time()
    other = {"A": ["configs/B.yaml"], "B": ["configs/A.yaml"]}[company]
    for use_transfer in (False, True):
        assoc = C.learn_transfer(transfer_sources({"transfer_from": other})) if use_transfer else None
        sub = tr
        P = C.classify_zero_shot(sub, tax, sub["supplier_cluster"], sub["item_key"].values, assoc,
                                 cfg.get("zero_shot", {}))
        pz = tax["leaf_id"].values[P.argmax(axis=1)]
        print(f"zero-shot labels hidden, transfer={use_transfer} ({time.time()-t0:.0f}s):",
              level_acc([lut[p] for p in pz], [lut[x] for x in tr["leaf_id"]], tr["spend_out"]))


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
