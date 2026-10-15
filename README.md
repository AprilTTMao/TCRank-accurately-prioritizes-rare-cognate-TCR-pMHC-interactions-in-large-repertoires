# TCRank

TCRank prioritizes candidate TCRs for experimental testing against a specified peptide-HLA target, using CDR3 or V-region sequences and a choice of predictors.

A web version is available at [URL TO BE ADDED].

## Installation

Install Docker and Python 3.9+, start Docker, and open a terminal in the extracted TCRank folder. The first run automatically loads the included Docker image, which contains `TCR_run.py`, the predictors and the files required for inference.

## Usage

CDR3 input:

```bash
python3 TCR_run.py --input examples/cdr3_smoke.csv --mode cdr3 --output results_cdr3 --predictors all
```

V-region input:

```bash
python3 TCR_run.py --input examples/vregion_smoke.csv --mode vregion --output results_vregion --predictors all
```

Replace the example CSV with your input file. To select predictors, replace `all` with their names, e.g. `--predictors PanPep TEINet ERGO-II`. Use `python3 TCR_run.py --list-predictors` to view available names. The output folder must be new or empty.

## Input

| Mode | CSV columns |
| --- | --- |
| CDR3 | `beta_cdr3,alpha_cdr3,peptide,mhc` (alpha is optional) |
| V region | `beta_v,alpha_v,peptide,mhc` (both chains required) |

Provide amino acid sequences and two-field human class-I HLA alleles, e.g. `HLA-A*02:01`.

## Output

- `ranking.csv`: all candidate pairs, with TCR sequences, peptide, HLA, scores and within-target ranks.
- `ranking.xlsx`: the same columns and results, with one sheet per peptide-HLA target (e.g. `GILGFVFTL_A0201`), ordered by rank. Full peptide and HLA values remain in the table.
- `predictor_scores.csv`: predictions from the selected tools.
- `summary.json`: run status and any unavailable predictions.
