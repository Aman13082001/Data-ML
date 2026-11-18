"""python -m spendcube --config configs/A.yaml [--data-dir data/A2] [--name A2]"""
import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description="Spend cube: ingest, clean, de-duplicate, classify.")
    ap.add_argument("--config", required=True, help="company config (YAML/JSON)")
    ap.add_argument("--data-dir", help="folder with the company's files (overrides config data_dir)")
    ap.add_argument("--name", help="output suffix, e.g. A2 -> <prefix>__A2.csv")
    ap.add_argument("--out-dir", help="output folder (default: outputs/)")
    ap.add_argument("--hide-labels", action="store_true", help="ignore labels: classify from the tree alone")
    a = ap.parse_args()
    run(a.config, a.data_dir, a.name, a.out_dir, a.hide_labels)


if __name__ == "__main__":
    main()
