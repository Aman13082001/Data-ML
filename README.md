# Spend cube pipeline (Cantill.ai Data & ML intern task)

This tool takes a company's raw purchase data and its category tree, and produces one row per input record. Each row says whether the record counts as spend (with a reason code if not), gives a clean supplier key and item key, the net spend in the reporting currency, and the category (L1 to L4) with a confidence.

**No LLM, no paid API, no internet at run time.** The pipeline uses pandas rules, fuzzy matching (rapidfuzz) and scikit-learn models. See [APPROACH.md](APPROACH.md) for the design and [DECISIONS.md](DECISIONS.md) for what I found in the data.

## Setup
```bash
pip install -r requirements.txt        # Python 3.10+
```
Put the task files under `data/`, which is git-ignored: the data is confidential and is never committed.
```
data/A/  erp_goods_receipts.csv  invoice_tool_export.csv  labels_A.csv  taxonomy_A.csv
data/B/  invoice_lines.csv  fx_rates_usd.csv  seed_labels.csv  taxonomy_B.csv
```

## The one command
```bash
python -m spendcube --config configs/A.yaml          # -> outputs/AK_Sharma__A.csv
python -m spendcube --config configs/B.yaml          # -> outputs/AK_Sharma__B.csv
```
New records from a known company use the same config with another data folder. The classifier still trains on the original labels (`labels.data_dir` in the config):
```bash
python -m spendcube --config configs/A.yaml --data-dir data/A2 --name A2 --out-dir submission
python -m spendcube --config configs/B.yaml --data-dir data/B2 --name B2 --out-dir submission
```
A new company needs only a new config:
```bash
python scripts/draft_config.py --company C --data-dir data/C --lines <lines file> --tree <tree file>
#   -> profiles every column, guesses roles, writes configs/C.yaml to review
python -m spendcube --config configs/C.yaml --out-dir submission    # no labels -> zero-shot classifier
```
Options: `--hide-labels` classifies from the tree alone, which is how I measured zero-shot accuracy on A.

## Re-running the blind test (for reviewers)
1. Put each data set in its own folder under `data/`. File names don't have to match the configs: a missing file is found by its column headers, and the tree, FX and label files are taken from the original `data/A` and `data/B` folders if they aren't resent.
   ```
   data/A/   data/B/      original kit (A2 and B2 train on these labels)
   data/A2/  data/B2/     new records, same layouts
   data/C/   C's invoice-line file + its category tree
   ```
2. Run the three commands:
   ```bash
   python -m spendcube --config configs/A.yaml      --data-dir data/A2 --name A2 --out-dir submission
   python -m spendcube --config configs/B.yaml      --data-dir data/B2 --name B2 --out-dir submission
   python -m spendcube --config submission/C.yaml   --data-dir data/C  --name C  --out-dir submission
   ```
3. This writes `submission/AK_Sharma__A2.csv`, `__B2.csv` and `__C.csv`. C has no labels, so it is classified zero-shot from its tree. The C config was written at the blind test, as the brief asks: it is the only file added after the build deadline, and it contains no code.

Runtime on a laptop CPU: A (65,549 records) takes about 1.5 min, and B (41,028) about 30 s.

## Output
`outputs/<prefix>__<name>.csv` has the columns `record_id, include, exclusion_reason, supplier_key, item_key, spend, l1, l2, l3, l4, confidence`.
`outputs/diagnostics/` (git-ignored, because it contains raw text) holds, per record, the currency, FX rate source, date flags, match rule and linked record, plus a summary JSON.

## Repository
```
spendcube/            the package
  config.py           YAML/JSON config + defaults merge, robust file reader (csv/tsv/xlsx, sniffed delimiter and encoding)
  ingest.py           per-source column mapping -> canonical frame; net of tax, FX, dates
  clean.py            number/date/rate parsing, OCR-aware text normalisation
  exclusions.py       status -> CANCELLED / NOT_APPROVED; NON_PROCUREMENT rules from config
  match.py            count once: cross-system matching + repeated exports
  entities.py         supplier and item de-duplication (union-find)
  taxonomy.py         category tree read at run time (level columns, path column, or parent/child)
  classify.py         supervised + zero-shot classifiers, hierarchical decoding
  textfeat.py         tokeniser, abbreviation expansion, concept lexicon
  pipeline.py         orchestration and output
configs/              defaults.yaml + one file per company
resources/            generic abbreviations and concept lexicon (no client category names)
scripts/evaluate.py   accuracy / de-dup metrics against the labels
scripts/draft_config.py   profiles a new company's files and drafts its config
```

## Tools used
Python, pandas, numpy, scikit-learn, scipy, rapidfuzz and PyYAML. I wrote the code with help from an AI coding assistant (Claude). The tool itself uses no LLM.
