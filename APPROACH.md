# APPROACH.md

**One command:** `python -m spendcube --config configs/<company>.yaml [--data-dir ...] [--name ...]`. No LLM and no paid API: classical data engineering and ML (pandas, rapidfuzz, scikit-learn), on a laptop CPU. **LLM cost per 1,000 lines: $0.**

## Pipeline
`config → ingest → exclusions → count once → supplier/item keys → classify → output`

1. **Ingest by config.** Each source's columns are mapped to one canonical schema (id, amount, gross/net, VAT rate or VAT amount, currency + rate column or FX table, dates, supplier name/id, description, item code, PO, invoice, status, extra classifier text). The reader sniffs the delimiter and encoding and also reads `.xlsx`. The tree is read at run time as level columns, one path column, or parent/child. A new company is a new YAML; `scripts/draft_config.py` profiles the file and drafts it.
2. **Clean.** Numbers in any format (`1,234.56`, `1.234,56`, `(12.00)`, `R 1 234`). VAT rates as `15`, `15%` or `0.15`, blank → default. Dates tried against a list of formats, each flagged ok / missing / unparseable / out of range. FX: the line's rate, else the table for the invoice month (nearest month if missing). A line rate more than 1.5× from the table rate is replaced, and USD stays 1:1.
3. **Exclusions,** in precedence CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE. Status words come from regexes in `defaults.yaml`. NON_PROCUREMENT uses generic rules (tax charges but not tax services, intercompany, royalties, interest, bank charges, payroll, partner billing …) plus company rules such as B's intercompany GL accounts.
4. **Count once.** Across systems, a secondary record (AP tool) is the same purchase as a primary record (ERP) when (a) it has the same PO + quantity + net amount, (b) it has the same PO and its amount is the sum of 2–3 receipts, or (c) it has the same quantity + amount and text similarity ≥ 85 after OCR normalisation (`R0D`→`ROD`, `5HAFTS`→`SHAFTS`). Within a system, rows equal on configured keys are one record exported twice. A duplicate inherits the keys and category of the record it duplicates.

## De-duplication
- **Suppliers:** normalise (accents, case, punctuation, legal suffixes, spaces: `OAK ENT MINING SA (PTY) LTD` → `OAKENTMINING`). Union-find over identical normalised names, shared vendor ids, and fuzzy ratio ≥ 92 (blocked on name prefix/suffix). The key is the most common spelling. The client's "outside the vendor master" bucket is a config option.
- **Items:** a material code is the item, and different codes are never merged. A line without a code joins the code whose (OCR-normalised, decimal-normalised) text it carries, when that text belongs to one code. Otherwise its normalised text is the key, merged fuzzily (≥ 94) only when the numbers are identical (size 6 ≠ size 8). Generic texts ("Product", "Parts") are keyed per supplier. A code used for more than 25 different texts is a placeholder, not an item.

## Classification
- **With labels (A, B):** char n-gram (2–5) and word TF-IDF, plus supplier, GL and categorical tokens, feed a logistic regression. An **item memory** lets an item the client already classified keep its class.
- **Unseen tree, no labels (C):** for every leaf, a weighted bag of its path words (leaf 1.0, parent 0.7, …). Each line's words are expanded with a generic abbreviation list (`VLV`→valve, `BRG`→bearing …, 250 entries) and plurals are stemmed. A generic concept lexicon (52 groups such as fasteners, electrical, freight, IT software, packaging, with common Spanish/Portuguese/French/German/Turkish words) adds shared tags to lines and leaves. The score is word cosine (with an IDF over leaves, so words shared by many categories count little) + character n-gram cosine (above a noise floor) + concept overlap. Then: softmax → smoothing over each supplier's and item's lines → **self-training** (a logistic regression on the most confident half of each predicted class, blended with the zero-shot scores, two rounds) → **hierarchical decoding** (best L1 by probability mass, then the best L2 inside it, … down to the leaf), because scoring is per level. The confidence is the leaf's probability blended with its L1 branch mass. Category names live only in the tree file, and the resources hold generic words only.

## Measured accuracy (spend-weighted, `scripts/evaluate.py`)
| | L1 | L2 | L3 |
|---|---|---|---|
| **A, labels hidden (zero-shot from the tree alone)** | **78.1%** | **61.3%** | **50.5%** |
| B, labels hidden | 97.5% | 92.0% | 29.0% |
| A supervised, 3-fold CV | 99.1% | 97.4% | 96.0% |
| B supervised, 3-fold CV on the 2,550 seed lines | 98.9% | 98.8% | 98.7% |

Line-weighted, A with labels hidden scores 66% / 40% / 32%. A's tree is organised by machine (continuous miner, shuttle car, LHD) rather than by part, so a part name often can't tell you the leaf. B's L3 is limited by aluminium coil lines whose text is a spec code (`0102X68.040,12OZ…`). Learning word→category-word associations from the *other* company's labels made both worse, so it is off by default (`transfer_from`).

**De-duplication on A (pairwise, vs the client's golden keys):** suppliers P = R = F1 = 1.000 (70/70); items P = 0.996, R = 0.995, F1 = 0.995. **Exclusions on B:** 100% agreement with the 3,000 seed labels. **Matching on A:** 23,552 tool lines linked to ERP receipts, 424 + 2 repeated exports, and 4,210 lines of spend found only in the tool.

## Runtime
A: 65,549 records in about 1 min (supervised). B: 41,028 in about 30 s. Zero-shot on 37k lines takes about 40 s. That is well inside 60 min for 25,000 lines.

## AI use
No LLM in the tool. I built the code with help from an AI coding assistant (Claude), and checked every rule against the data myself.
