"""Classify included lines into the company's tree (lowest level) with a confidence.

Two modes, chosen by whether labels are available:
  supervised  - TF-IDF (char + word) + supplier/categorical tokens -> logistic
                regression, plus an item memory (item_key seen in the labels).
  zero-shot   - no labels: score each line against every leaf of the tree using
                (1) word overlap with the path names (after abbreviation
                expansion and stemming), (2) character n-gram similarity,
                (3) concept tags from a generic lexicon, (4) word->category-word
                associations learned from *other* companies' labelled data;
                then self-train a classifier on the confident lines and smooth
                over suppliers and items.
No category names appear in this file: the tree is data.
"""
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from .clean import compact, norm_text
from .textfeat import lexicon


def line_text(df):
    return (df["description"].fillna("") + " " + df["text_extra"].fillna("")).str.strip()


def _softmax(s, temp):
    z = (s - s.max(axis=1, keepdims=True)) / temp
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def hierarchical_decode(P, tax):
    """Choose level by level: the L1 with the most probability mass, then the
    best L2 inside it, ... down to a leaf. Scoring is per level, so this beats a
    flat argmax when the leaf is uncertain but the branch is not.
    Returns (leaf index, confidence = probability of the chosen leaf's branch
    at the deepest level, i.e. the chosen leaf's own probability renormalised
    inside its chosen parents)."""
    paths = list(tax["path"])
    n = P.shape[0]
    alive = np.ones((n, len(paths)), bool)
    depth = max(len(p) for p in paths)
    for lvl in range(depth):
        names = np.array([p[lvl] if lvl < len(p) else p[-1] for p in paths])
        uniq = np.unique(names)
        mass = np.stack([np.where(alive[:, names == u], P[:, names == u], 0).sum(axis=1) for u in uniq], axis=1)
        pick = uniq[mass.argmax(axis=1)]
        alive &= names[None, :] == pick[:, None]
    best = np.where(alive, P, -1).argmax(axis=1)
    conf = P[np.arange(n), best]
    # confidence: mass of the chosen L1 branch blended with the leaf's own probability
    l1 = np.array([p[0] for p in paths])
    l1_mass = np.array([P[i, l1 == l1[best[i]]].sum() for i in range(n)])
    return best, 0.5 * conf + 0.5 * l1_mass


# ===================================================================== supervised
class Supervised:
    def __init__(self, C=10.0, max_iter=300):
        lex = lexicon()
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    sublinear_tf=True, max_features=200000, preprocessor=norm_text)
        self.word = TfidfVectorizer(analyzer=lambda s: lex.tokens(s) + lex.concepts(lex.tokens(s)),
                                    min_df=1, sublinear_tf=True)
        self.tok = TfidfVectorizer(analyzer=lambda s: s.split(), min_df=1, sublinear_tf=True, use_idf=False)
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def _X(self, text, toks, fit=False):
        f = (lambda v, x: v.fit_transform(x)) if fit else (lambda v, x: v.transform(x))
        return sparse.hstack([f(self.char, text), f(self.word, text), f(self.tok, toks) * 0.7]).tocsr()

    def fit(self, text, toks, y, w=None):
        X = self._X(list(text), list(toks), fit=True)
        self.model.fit(X, y, sample_weight=w)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, text, toks):
        return self.model.predict_proba(self._X(list(text), list(toks)))


def tokens_col(df, supplier_key):
    sup = "SUP_" + supplier_key.fillna("").map(lambda s: compact(norm_text(s)))
    cat = df["cat_extra"].fillna("").astype(str).map(
        lambda s: " ".join("CAT_" + compact(norm_text(x)) for x in s.split(" ") if x))
    return (sup + " " + cat).str.strip()


# ===================================================================== zero-shot
def _leaf_bags(tax, level_weights):
    lex = lexicon()
    bags = []
    for path in tax["path"]:
        bag = Counter()
        d = len(path)
        for i, name in enumerate(path):
            # level_weights are counted from the leaf upwards: the leaf name is
            # the most specific, each level above it a little less
            w = level_weights[min(d - 1 - i, len(level_weights) - 1)]
            toks = lex.tokens(name)
            for t in set(toks):
                bag[t] = max(bag[t], w)
            for g in lex.concepts(toks):
                bag["@" + g] = max(bag["@" + g], w)
        bags.append(dict(bag))
    return bags


def _line_bags(df, supplier_names, sup_weight):
    lex = lexicon()
    bags = []
    for txt, sup in zip(line_text(df), supplier_names):
        toks = lex.tokens(txt)
        bag = Counter()
        for t in toks:
            bag[t] += 1.0
        for g in lex.concepts(toks):
            bag["@" + g] += 1.0
        if sup_weight:
            st = lex.tokens(sup)
            for t in st:
                bag[t] += sup_weight
            for g in lex.concepts(st):
                bag["@" + g] += sup_weight
        bags.append(dict(bag))
    return bags


def learn_transfer(sources):
    """sources: list of (texts, paths). Learn P(category-name word | line word)
    across other companies' labelled data. Returns {line_tok: {name_tok: weight}}."""
    lex = lexicon()
    co = defaultdict(Counter)
    tcount = Counter()
    ncount = Counter()
    n_lines = 0
    for texts, paths in sources:
        for txt, path in zip(texts, paths):
            if not path:
                continue
            nt = set(t for name in path for t in lex.tokens(name))
            lt = set(lex.tokens(txt))
            if not lt or not nt:
                continue
            n_lines += 1
            for n in nt:
                ncount[n] += 1
            for t in lt:
                tcount[t] += 1
                co[t].update(nt)
    assoc = {}
    for t, c in co.items():
        if tcount[t] < 3:
            continue
        row = {}
        for n, k in c.items():
            p = k / tcount[t]
            lift = p / (ncount[n] / n_lines)
            if lift > 2.0 and k >= 2:
                row[n] = p * math.log(lift)
        if row:
            assoc[t] = row
    return assoc


def zero_shot_scores(df, tax, supplier_names, transfer=None, zc=None):
    zc = zc or {}
    lw = zc.get("level_weights", [1.0, 0.7, 0.45, 0.3])
    leaf_b = _leaf_bags(tax, lw)
    line_b = _line_bags(df, supplier_names, zc.get("supplier_weight", 0.5))
    dv = DictVectorizer()
    dv.fit(leaf_b)
    L = dv.transform(leaf_b)
    X = dv.transform(line_b)
    # idf over leaves: words shared by many categories carry little information
    df_leaf = np.asarray((L > 0).sum(axis=0)).ravel()
    idf = np.log((1 + L.shape[0]) / (1 + df_leaf)) + 1.0
    D = sparse.diags(idf)
    s_word = normalize(X @ D) @ normalize(L @ D).T
    s_word = s_word.toarray()

    # character n-grams: catches abbreviations not in the lexicon (HYD ~ HYDRAULIC)
    names = [" ".join(p[-2:]) for p in tax["path"]]
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=norm_text, sublinear_tf=True)
    cv.fit(names + line_text(df).tolist())
    s_char = (cv.transform(line_text(df)) @ cv.transform(names).T).toarray()
    # weak character overlap is noise, not evidence: it would decide lines that
    # have no real signal (those are better left to supplier/item smoothing)
    s_char = np.maximum(0.0, s_char - zc.get("char_floor", 0.1))

    s = zc.get("w_word", 1.0) * s_word + zc.get("w_char", 0.35) * s_char

    if transfer:
        lex = lexicon()
        leaf_names = [Counter(t for name in p for t in lex.tokens(name)) for p in tax["path"]]
        dvn = DictVectorizer()
        Ln = dvn.fit_transform(leaf_names)
        vocab = dvn.vocabulary_
        rows, cols, vals = [], [], []
        for i, txt in enumerate(line_text(df)):
            acc = defaultdict(float)
            for t in set(lex.tokens(txt)):
                for n, w in transfer.get(t, {}).items():
                    if n in vocab:
                        acc[n] += w
            for n, w in acc.items():
                rows.append(i)
                cols.append(vocab[n])
                vals.append(w)
        T = sparse.csr_matrix((vals, (rows, cols)), shape=(len(df), len(vocab)))
        dfn = np.asarray((Ln > 0).sum(axis=0)).ravel()
        Dn = sparse.diags(np.log((1 + Ln.shape[0]) / (1 + dfn)) + 1.0)
        s_tr = (normalize(T @ Dn) @ normalize(Ln @ Dn).T).toarray()
        s = s + zc.get("w_transfer", 0.6) * s_tr
    return s


def _group_smooth(P, keys, weight, spend=None):
    if not weight:
        return P
    keys = pd.Series(keys).fillna("").astype(str).values
    w = np.ones(len(keys)) if spend is None else np.log1p(np.abs(spend)) + 1.0
    df = pd.DataFrame(P * w[:, None])
    df["k"] = keys
    sums = df.groupby("k").sum()
    cnt = pd.Series(w).groupby(keys).sum()
    G = (sums.loc[keys].values) / cnt.loc[keys].values[:, None]
    blank = keys == ""
    G[blank] = P[blank]
    return P + weight * G


def classify_zero_shot(df, tax, supplier_key, item_key, transfer=None, zc=None):
    zc = zc or {}
    s = zero_shot_scores(df, tax, df["supplier_name"].fillna("").tolist(), transfer, zc)
    P0 = _softmax(s, zc.get("temperature", 0.05))
    no_signal = s.max(axis=1) <= 1e-9
    P = P0.copy()
    text = line_text(df)
    toks = tokens_col(df, supplier_key)
    for _ in range(zc.get("self_train_rounds", 2)):
        P = _group_smooth(P, item_key, zc.get("item_smooth", 1.0))
        P = _group_smooth(P, supplier_key, zc.get("supplier_smooth", 0.5))
        P = P / P.sum(axis=1, keepdims=True)
        top = P.argmax(axis=1)
        conf = P.max(axis=1)
        # confident pseudo-labels: top share per predicted class
        keep = np.zeros(len(df), bool)
        q = zc.get("pseudo_label_share", 0.5)
        for c in np.unique(top):
            idx = np.where((top == c) & ~no_signal)[0]
            if len(idx) == 0:
                continue
            thr = np.quantile(conf[idx], 1 - q)
            keep[idx[conf[idx] >= thr]] = True
        if keep.sum() < 50 or len(np.unique(top[keep])) < 2:
            break
        m = Supervised(C=zc.get("self_train_C", 3.0))
        m.fit(text[keep], toks[keep], top[keep], w=conf[keep])
        Pst = np.zeros_like(P)
        Pst[:, m.classes_] = m.predict_proba(text, toks)
        P = zc.get("zs_weight", 0.5) * P0 + (1 - zc.get("zs_weight", 0.5)) * Pst
    P = _group_smooth(P, item_key, zc.get("item_smooth", 1.0))
    P = P / P.sum(axis=1, keepdims=True)
    return P


def classify_supervised(df, train_text, train_toks, train_y, supplier_key, item_key,
                        train_items=None, sc=None):
    sc = sc or {}
    m = Supervised(C=sc.get("C", 10.0))
    m.fit(train_text, train_toks, train_y)
    text = line_text(df)
    toks = tokens_col(df, supplier_key)
    P = m.predict_proba(text, toks)
    classes = list(m.classes_)
    # item memory: an item the client already classified keeps its class
    if train_items is not None and sc.get("item_memory", True):
        tab = pd.crosstab(pd.Series(list(train_items), name="i"), pd.Series(list(train_y), name="y"))
        pur = tab.max(axis=1) / tab.sum(axis=1)
        best = tab.idxmax(axis=1)
        good = pur >= sc.get("item_memory_purity", 0.8)
        lut = {i: (best[i], pur[i]) for i in pur[good].index}
        col = {c: j for j, c in enumerate(classes)}
        for r, it in enumerate(item_key):
            hit = lut.get(it)
            if hit is not None and hit[0] in col:
                P[r] = P[r] * 0.3
                P[r, col[hit[0]]] += 0.7 * hit[1]
    return P, classes
