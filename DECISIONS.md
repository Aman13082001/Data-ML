# DECISIONS.md: what the data showed and what I decided

Every rule below lives in a config file (`configs/*.yaml`), not in code, unless it says "generic". A new company needs a new config.

## Company A: mining, SAP goods receipts + AP invoice tool (ZAR)

### What I found
| Finding | Evidence |
|---|---|
| The tool file is mostly a re-keyed copy of ERP receipts | 16,743 tool lines match an ERP line on PO + quantity + net amount exactly |
| PO references are typed 5 ways | `4510062123`, `PO 4510062123`, `PO#4510062123`, `4510062123/10.0`, blank (5,199) |
| Invoice numbers are typed 5+ ways | `IN0701589`, `SI-2024-018414`, `TAX INV 132077`, `INV/2024/211523`, `825128` |
| Tool line text is OCR'd | `R0D DRIL`, `5HAFTS`, `B1T RCK`, `LA8R&EQ`, `HO5E`, `the Company` vs `THE COMPANY` |
| Supplier names are re-typed | `OAK ENT MINING SA (PTY) LTD`, `Prahaven (Pty) Ltd`, `PEL WICK (PTY) LTD`, `COR ORA` |
| Gross amounts use thousands separators | `5,266.63` next to `5266.63` |
| The VAT rate is written 4 ways | `15`, `0.15`, `15%`, blank (6,542 lines); `0` on 1,580 lines (zero-rated) |
| Invoice dates are in 3 formats, 65% blank | `2024-03-29`, `16/05/2024`, `08-May-24` |
| ERP `posting_date` is blank on 24,370 lines | Not needed for spend; flagged |
| `value_zar` is ZAR even when `document_currency` is EUR/CHF/MZN (16 lines) | Unit cost × qty = value_zar |
| Some tool invoices were exported twice | Consecutive record ids, same invoice digits + qty + gross, only OCR noise differs |
| Some tool invoices bill several receipts at once | e.g. qty 3,000 = 2,000 + 1,000 receipts on the same PO |
| ERP has 434 negative lines (receipt reversals) and 509 lines valued 0.01 | |

### What I decided
1. **Net of tax.** I use ERP `value_zar` as is. For a tool line, net = gross / (1 + rate). `15`, `15%` and `0.15` all mean 15%, blank means 15%, and `0` means zero-rated (net = gross).
2. **Count once (tool vs ERP).** A tool line counts as the same purchase as an ERP line, in order of evidence:
   - (a) same PO number (digits extracted from any format), same quantity, and the same net amount within 0.5%;
   - (b) same PO, and the amount equals the sum of 2 or 3 receipts on that PO (one invoice for several receipts);
   - (c) same quantity, the same net amount, and line text ≥ 85% similar after OCR normalisation. This covers a missing or mistyped PO. On the 5,199 tool lines without a PO, the best text similarity is bimodal: 4,234 score ≥ 85 and almost all the rest score < 50.

   The ERP record counts, and the tool record is `DUPLICATE`.
3. **Exported twice.** Tool lines with the same invoice-number digits, quantity and gross amount are one record. The first copy counts and the others are `DUPLICATE`. If any copy matched an ERP line, all copies are `DUPLICATE`.
4. **Spend only in the tool (4,210 lines)** counts when the line is Approved or Paid: 2,898 have a PO that isn't in the ERP file, and 823 have no PO and no text/amount match. 489 have a PO that is in the ERP file but a quantity and amount that match no receipt, or sum of 2–3 receipts, on it. Their unit price equals the PO's, so I treat them as receipts missing from the ERP extract, not as duplicates.
5. **Status.** Paid and Approved count. Cancelled → `CANCELLED`. Rejected and On Hold → `NOT_APPROVED`. Status is checked before matching, so a cancelled tool line keeps `CANCELLED` even when it also matches an ERP receipt. The ERP receipt still counts: the goods were received.
6. **ERP reversals and 0.01 lines are included.** A reversal and its original net to zero, which is the correct spend. Excluding one of the pair would distort the total.
7. **Dates** are parsed against a list of formats (day-first). Each record is flagged `MISSING`, `UNPARSEABLE` or `OUT_OF_RANGE` in `outputs/diagnostics/`. A bad date never removes spend.
8. **Client conventions found in `labels_A`, followed because the keys are the client's:**
   - **Suppliers:** 120 supplier names share one golden key. They are exactly the names whose ERP records mostly have no vendor id: suppliers outside the vendor master. The rule "a cluster with vendor ids on fewer than 50% of its ERP records is the unmastered bucket" reproduces the 70 golden suppliers exactly (pairwise F1 1.000). Otherwise one key per real supplier: legal suffixes removed (PTY, LTD, CC, SA …), spaces and punctuation removed, then fuzzy match ≥ 92, plus shared vendor ids.
   - **Items:** golden items exist only for lines with a material number (they're blank on the 15,816 free-text and service lines). Catalogue and free-text buys with a material number but no material master data (no `material_type`) share one golden key, a rule that covers 2,659 of its 2,661 lines. Otherwise **a material code is the item**. Two codes are never merged, even when the text is identical (`POWDER;STNEDST` exists under several codes, and the client keeps them apart). A line without a code joins a code whose text it carries, when that text belongs to exactly one code. Item text is `material_short_text` only, because `service_short_text` holds warranty and repair notes that would split an item. Pairwise F1 0.995.
9. Tool lines inherit the supplier, item and category of the ERP line they duplicate.

## Company B: metal packaging, 8 ERPs, 25 currencies (USD)

### What I found
| Finding | Evidence |
|---|---|
| No status column | Every line is a posted AP line, so nothing is CANCELLED/NOT_APPROVED by status |
| Intercompany is visible in the GL description | `IC05`, `IC06` … `IC70`, `IC Payables`, `I/C Payable`, `Intercompany - CZ`; all NON_PROCUREMENT in the seed labels |
| Tax lines are booked as invoice lines | `State & Local Taxes -`, `Sales Tax Expense -`, `GST Input Tax -`, `Accrued Sales & Use Tax` |
| 1,279 USD lines carry a `currency_rate` of about 17.5 (an MXN rate) | The rule says USD is already USD |
| 64 non-USD lines have a rate 1.5× to 20× away from the month's table rate | EUR lines with ~18–20 (also an MXN-like rate), CHF, ARS |
| 3,343 non-USD lines have no rate | CAD, MMK, THB have none at all |
| The FX table has gaps | e.g. AED has no 2024-02 |
| 613 lines are exact copies of another line except `line_id` (1,066 lines in repeated groups) | Same ERP, invoice, line number, amount, dates, everything |
| Invoice dates are ISO but some years are impossible | 2018, 2020, 2027 |
| 16,695 descriptions are blank or whitespace | |
| `MAT-1000007` is on 10,484 lines with hundreds of different texts | A placeholder, not an item |
| Supplier is a vendor id only | No names, so no evidence to merge ids across ERPs |
| 3 `RCN` lines have no amount | RCN is in the FX table at 7.08 (looks like CNY) |

### What I decided
1. **NON_PROCUREMENT** = intercompany GL description (config regex), plus tax descriptions (generic rule in `defaults.yaml`). A tax line is excluded only if it is a tax *charge*, not a tax *service*: `PayTech Tax Consultant`, `Blanket PO for Tax work` and `Assistance with Property Tax` stay as spend. `Tax Services and Garnishments` is excluded, because garnishments are payroll deductions, and the seed agrees. **Result: 100% agreement with the seed labels on include/exclude (3,000 lines).**
2. **FX.** USD lines always stay 1:1. A non-USD line uses its own `currency_rate` (units per USD, so USD = amount / rate). With no rate, I use the table for the invoice month. If the invoice date is missing or impossible, I use the posting month, and with no table row that month, the nearest month for that currency. A given rate more than 1.5× away from the table rate is a data error, so I use the table rate instead (64 lines, flagged `RATE_REPLACED_BY_TABLE`).
3. **Re-exports** (all columns equal except the id) → the first copy counts, and the rest are `DUPLICATE` (604 records; the other 9 copies were already NON_PROCUREMENT).
4. **Credit notes and reversals** (negative amounts, 833 lines) are included and net against their originals. The seed labels include negative lines too.
5. **Blank descriptions** are still classified, from the GL account description, GL account, supplier type, PO type and vendor history.
6. **Supplier key = vendor id**, normalised. Items: a material code is the item, except a code used for more than 25 different texts, which is treated as a placeholder and falls back to the text.
7. **Impossible dates** are flagged `OUT_OF_RANGE` and replaced by the posting date for the FX month. Spend is kept.
8. 3 lines with no amount → spend 0, flagged `NO_AMOUNT_OR_FX`. They stay included, because the brief gives no exclusion code for "no amount".

## Generic decisions (all companies, `configs/defaults.yaml`)
- **Status words:** `paid / approved / posted / cleared / closed …` count. `cancel / void / reversed` → CANCELLED. `rejected / hold / disputed / submitted / pending / draft / parked / open …` → NOT_APPROVED. An unknown status counts as NOT_APPROVED, because only approved or paid counts. A blank status counts, because a system without a status column only holds posted lines.
- **NON_PROCUREMENT words:** taxes (unless tax *services*), intercompany, royalties, dividends, bank and finance charges, interest, loan repayments, partner billings, revenue or profit share, penalties, garnishments, salaries and payroll (unless payroll *services/software*), and payees that are tax authorities.
- **The precedence of exclusion codes:** CANCELLED > NOT_APPROVED > NON_PROCUREMENT > DUPLICATE.
- **Confidence on excluded rows** = the confidence of the rule (0.95 for status and keyword rules, 0.97 for exact PO/qty/amount matches and re-exports, 0.85 for text-based matches).
- **When account code, supplier and text disagree,** text decides. The account code and supplier are features in the classifier, and the supplier also smooths the zero-shot result. A clear description outranks a generic account.
- **Excluded rows have blank l1–l4.** They are not spend, so they get no category.
