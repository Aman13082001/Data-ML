"""Load a company config (YAML/JSON) and merge it over the shared defaults."""
import copy
import json
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULTS = os.path.join(REPO, "configs", "defaults.yaml")


def _read(path):
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return yaml.safe_load(f) or {}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path, data_dir=None):
    cfg = deep_merge(_read(DEFAULTS), _read(path))
    cfg["_config_path"] = os.path.abspath(path)
    # data_dir: CLI override > config value (relative to repo root)
    base = cfg.get("data_dir", ".")
    base = base if os.path.isabs(base) else os.path.join(REPO, base)
    d = data_dir or base
    if not os.path.isabs(d):
        d = os.path.join(REPO, d)
    cfg["data_dir"] = d
    cfg["_build_data_dir"] = base       # where the original (build) files live
    return cfg


def _header(path):
    try:
        return set(read_table(path).columns.str.strip()) if os.path.getsize(path) < 2e8 else set()
    except Exception:
        return set()


def resolve_file(cfg, filename, required_cols=(), fallback=True):
    """Find a configured file. Order: <data_dir>/<filename>; a file in data_dir
    whose header holds all required columns (new extracts may be renamed);
    then the build data dir (tree / FX / labels are not resent with new records)."""
    p = data_path(cfg, filename)
    if p and os.path.exists(p):
        return p
    d = cfg["data_dir"]
    if required_cols and os.path.isdir(d):
        hits = []
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp) and f.lower().endswith((".csv", ".txt", ".tsv", ".xlsx", ".xls")):
                h = _header(fp)
                if set(required_cols) <= h:
                    hits.append(fp)
        if len(hits) == 1:
            print(f"  note: '{filename}' not found; using {os.path.basename(hits[0])} (matching columns)")
            return hits[0]
    if fallback:
        for alt in [cfg.get("_build_data_dir"), (cfg.get("labels") or {}).get("data_dir")]:
            if not alt:
                continue
            alt = alt if os.path.isabs(alt) else os.path.join(REPO, alt)
            q = os.path.join(alt, filename)
            if os.path.exists(q):
                print(f"  note: '{filename}' not in {d}; using {q}")
                return q
    raise FileNotFoundError(f"{filename} not found in {d}")


def read_table(path, **opts):
    """Read CSV/TSV/TXT/XLSX as strings. Delimiter and encoding are sniffed
    unless given (opts: sep, encoding, sheet)."""
    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=opts.get("sheet", 0), dtype=str)
    encs = [opts["encoding"]] if opts.get("encoding") else ["utf-8-sig", "cp1252", "latin-1"]
    last = None
    for enc in encs:
        try:
            sep = opts.get("sep")
            if sep is None:
                with open(path, encoding=enc) as f:
                    head = f.readline()
                sep = max([",", ";", "\t", "|"], key=head.count)
            return pd.read_csv(path, dtype=str, sep=sep, encoding=enc, keep_default_na=False, na_values=[""])
        except UnicodeDecodeError as e:
            last = e
    raise last


def data_path(cfg, filename):
    if filename is None:
        return None
    return filename if os.path.isabs(filename) else os.path.join(cfg["data_dir"], filename)
