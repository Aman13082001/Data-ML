"""Supplier and item de-duplication with union-find over several kinds of evidence."""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .clean import compact, norm_text


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        p = self.p
        p.setdefault(x, x)
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _fuzzy_pairs(keys, threshold, block=3, min_len=5, scorer=fuzz.ratio, same_digits=False):
    """Pairs of distinct strings with similarity >= threshold, compared inside
    blocks sharing the first `block` characters (plus a second pass on the last
    characters, so a typo at the start is still caught). With same_digits the
    two strings must contain the same numbers (size 6 != size 8)."""
    keys = [k for k in set(keys) if len(k) >= min_len]
    pairs = []
    digits = (lambda s: re.sub(r"\D", "", s)) if same_digits else None
    for keyfn in (lambda k: k[:block], lambda k: k[-block:]):
        blocks = defaultdict(list)
        for k in keys:
            blocks[keyfn(k)].append(k)
        for grp in blocks.values():
            if len(grp) < 2:
                continue
            m = process.cdist(grp, grp, scorer=scorer, score_cutoff=threshold, workers=-1)
            ii, jj = np.nonzero(m)
            for i, j in zip(ii, jj):
                if i < j and (digits is None or digits(grp[i]) == digits(grp[j])):
                    pairs.append((grp[i], grp[j]))
    return pairs


# ------------------------------------------------------------------ suppliers
def supplier_base(name, legal_rx):
    t = norm_text(name, ocr=False)
    t = re.sub(r"[^A-Z0-9 ]", " ", t)
    t = legal_rx.sub(" ", " " + t + " ")
    return compact(t)


def supplier_keys(df, cfg):
    sc = cfg.get("supplier_dedup", {})
    legal = sc.get("legal_suffixes", [])
    legal_rx = re.compile(r"(?<= )(" + "|".join(re.escape(x) for x in legal) + r")(?= )") if legal else re.compile(r"$^")
    names = df["supplier_name"].fillna("").astype(str)
    ids = df["supplier_id"].fillna("").astype(str).str.strip()
    base = names.map(lambda n: supplier_base(n, legal_rx))
    # no name: fall back to the vendor id
    node = np.where(base != "", "N:" + base, np.where(ids != "", "I:" + ids, "UNKNOWN"))
    node = pd.Series(node, index=df.index)
    uf = UnionFind()
    for n in node.unique():
        uf.find(n)
    if sc.get("merge_on_vendor_id", True):
        both = (base != "") & (ids != "")
        for (b, i) in set(zip(base[both], ids[both])):
            uf.union("N:" + b, "I:" + i)
    thr = sc.get("fuzzy_threshold", 92)
    for a, b in _fuzzy_pairs(base[base != ""].unique().tolist(), thr, block=sc.get("block", 2)):
        uf.union("N:" + a, "N:" + b)
    root = node.map(uf.find)
    # readable key: the most common raw spelling in each cluster
    label = {}
    for r, g in names.groupby(root):
        c = Counter(x for x in g if x)
        label[r] = (c.most_common(1)[0][0] if c else r).strip().upper()
    key = root.map(label)
    cluster = key.copy()        # the real supplier, kept for the classifier
    # optional client convention: suppliers outside the vendor master (their
    # records mostly carry no vendor id) share one key
    um = sc.get("unmastered")
    if um:
        srcs = [s["name"] for s in cfg["sources"] if (s.get("supplier") or {}).get("id_column")]
        m = df["source"].isin(srcs)
        share = (ids[m] != "").groupby(root[m]).mean()
        bucket = set(share[share < um.get("id_share_below", 0.5)].index)
        if um.get("include_clusters_without_id_source", True):
            bucket |= set(root.unique()) - set(share.index)
        key = key.where(~root.isin(bucket), um.get("key", "UNMASTERED SUPPLIER"))
    return key, cluster


# ------------------------------------------------------------------ items
def item_text_key(desc):
    t = norm_text(desc)
    t = re.sub(r"\bP/?N\b", "PN", t)
    return compact(t)


def item_keys(df, cfg, supplier_key):
    ic = cfg.get("item_dedup", {})
    generic = set(compact(norm_text(g)) for g in ic.get("generic_descriptions", []))
    code = df["item_code"].fillna("").astype(str).str.strip()
    txt = df["item_text"].fillna("").map(item_text_key)
    is_generic = txt.isin(generic) | (txt.str.len() < ic.get("min_text_len", 3))
    # a code shared by many different texts is a placeholder (e.g. 'non-stock'), not an item
    ntexts = txt[~is_generic].groupby(code[~is_generic]).nunique()
    placeholder = set(ntexts[ntexts > ic.get("max_texts_per_code", 25)].index) - {""}
    code = code.where(~code.isin(placeholder), "")
    uf = UnionFind()
    node = pd.Series(np.where(code != "", "C:" + code, "T:" + txt), index=df.index)
    # generic text with no code: the item is only knowable per supplier
    gen_nodes = (code == "") & is_generic
    node[gen_nodes] = "G:" + supplier_key[gen_nodes].astype(str) + "|" + txt[gen_nodes]
    for n in node.unique():
        uf.find(n)
    # A code is the item. Different codes are never merged (the same text can
    # legitimately sit under two codes). A row without a code joins the code
    # whose text it carries, when that text belongs to exactly one code.
    ok = (code != "") & ~is_generic
    t2c = pd.Series(code[ok].values, index=txt[ok].values).groupby(level=0).agg(lambda s: set(s))
    coded_text = {t for t, cs in t2c.items()}
    for t, cs in t2c.items():
        if len(cs) == 1:
            uf.union("C:" + next(iter(cs)), "T:" + t)
    thr = ic.get("fuzzy_threshold", 92)
    if thr and thr < 100:
        allk = list(set(txt[~is_generic].unique()))
        for a, b in _fuzzy_pairs(allk, thr, block=3, min_len=ic.get("fuzzy_min_len", 8), same_digits=True):
            if a in coded_text and b in coded_text:
                continue        # two coded texts: keep their codes apart
            amb = (a in t2c and len(t2c[a]) > 1) or (b in t2c and len(t2c[b]) > 1)
            if not amb:
                uf.union("T:" + a, "T:" + b)
    root = node.map(uf.find)
    # optional client conventions: rows meeting a rule share one fixed key
    for rule in ic.get("bucket_rules", []):
        m = pd.Series(True, index=df.index)
        for cond in rule["when"]:
            s = df.get("src." + cond["column"], pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
            if "present" in cond:
                m &= (s != "") if cond["present"] else (s == "")
            if "pattern" in cond:
                m &= s.str.contains(cond["pattern"], case=False, regex=True)
        if rule.get("sources"):
            m &= df["source"].isin(rule["sources"])
        root = root.where(~m, rule["key"])
    return root
