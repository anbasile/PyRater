# Command line

```
pyrater [OPTIONS] INPUT_FILE
```

A single flat command — no subcommand word — matching the paper's own
Listing 4.

| Option | Default | Meaning |
|---|---|---|
| `-m, --model` | *(required)* | Registered model name |
| `--format` | `long` | `long`, `wide`, or `snow2008` |
| `--item-col` / `--annotator-col` / `--label-col` | `item_id` / `annotator_id` / `label` | Column names for long format |
| `--gold` | — | CSV of `item_id,label` for post-hoc alignment |
| `--chains` / `--warmup` / `--iters` / `--seed` | `4` / `1000` / `1000` / none | MCMC sampling |
| `--align` / `--no-align` | align | Label-switching correction |
| `--check-rhat` / `--no-check-rhat` | check | Convergence check |
| `--rhat-threshold` | `1.01` | |
| `-o, --output-dir` | `./pyrater-output` | |
| `--plot` / `--no-plot` | no-plot | Write PNGs alongside the CSVs |
| `--list-models` | | Print available models and exit |
| `--describe NAME` | | Print a model's description and exit |
| `--version` | | |

## Output files

`labels.csv`, `labels_proba.csv`, `difficulty.csv`, `credibilities.csv` (or
`credibilities.txt` explaining why, if the input data has no annotator
identifiers), `convergence.json`, and `plots/*.png` if `--plot` was passed.

**Exit codes:** `0` on success (including a convergence warning — that's
not a CLI failure); `1` on a data/model error (unknown model, bad file);
`2` on an argument-parsing error.

## Example

```bash
pyrater -m d&s annotations.csv --gold gold.csv -o results/ --plot
cat results/labels.csv
```
