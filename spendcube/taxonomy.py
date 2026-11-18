"""Category tree read at run time. Nothing about any tree lives in code.

Supported layouts (config `taxonomy:`):
  level_columns: [L1, L2, L3, L4]       one row per leaf, a column per level (default: all columns)
  path_column: Path, separator: ' > '   one row per leaf, the full path in one column
  parent/child: {id_column, parent_column, name_column}   an adjacency list
"""
import pandas as pd

from .config import read_table, resolve_file


def _paths_from_adjacency(t, tc):
    ids = t[tc["id_column"]].astype(str).str.strip()
    par = t[tc["parent_column"]].fillna("").astype(str).str.strip()
    name = dict(zip(ids, t[tc["name_column"]].astype(str).str.strip()))
    parent = dict(zip(ids, par))
    children = set(p for p in par if p)
    paths = []
    for i in ids:
        if i in children:
            continue
        p, cur, seen = [], i, set()
        while cur and cur in name and cur not in seen:
            seen.add(cur)
            p.append(name[cur])
            cur = parent.get(cur, "")
        paths.append(tuple(reversed(p)))
    return paths


def load_taxonomy(cfg):
    tc = cfg["taxonomy"]
    t = read_table(resolve_file(cfg, tc["file"], tc.get("level_columns") or ([tc["path_column"]] if tc.get("path_column") else [])),
                   sep=tc.get("sep"), encoding=tc.get("encoding"))
    t.columns = [str(c).strip() for c in t.columns]
    t = t.fillna("")
    if tc.get("path_column"):
        sep = tc.get("separator", ">")
        paths = [tuple(x.strip() for x in str(v).split(sep) if x.strip()) for v in t[tc["path_column"]]]
        n_levels = max(len(p) for p in paths)
    elif tc.get("id_column"):
        paths = _paths_from_adjacency(t, tc)
        n_levels = max(len(p) for p in paths)
    else:
        cols = tc.get("level_columns") or list(t.columns)
        t = t[cols].apply(lambda s: s.astype(str).str.strip())
        t = t[(t != "").any(axis=1)].drop_duplicates().reset_index(drop=True)
        paths = [tuple(x for x in row if x) for row in t.itertuples(index=False)]
        n_levels = len(cols)
    paths = [p for p in paths if p]
    # keep only true leaves (a path that is a prefix of another is not a leaf)
    pset = set(paths)
    leaves = [p for p in dict.fromkeys(paths) if not any(q != p and q[:len(p)] == p for q in pset)]
    tax = pd.DataFrame({"path": leaves})
    tax["leaf_id"] = tax["path"].map(lambda p: " > ".join(p))
    tax["depth"] = tax["path"].map(len)
    tax.attrs["n_levels"] = n_levels
    return tax


def path_from_labels(labels_df, level_cols, tax):
    """Map label rows onto taxonomy leaves (exact names; blank levels ignored)."""
    by_path = {p: lid for p, lid in zip(tax["path"], tax["leaf_id"])}
    out = []
    for row in labels_df[level_cols].fillna("").astype(str).itertuples(index=False):
        p = tuple(x.strip() for x in row if x and x.strip())
        lid = by_path.get(p)
        if lid is None:
            # tolerate a label that stops above the leaf level: take it only if unique
            cands = [l for q, l in by_path.items() if q[:len(p)] == p] if p else []
            lid = cands[0] if len(cands) == 1 else None
        out.append(lid)
    return pd.Series(out, index=labels_df.index)
