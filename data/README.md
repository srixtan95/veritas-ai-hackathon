# Data

The project uses the Statlog German Credit dataset, UCI Machine Learning
Repository dataset **144**. The training pipeline retrieves the data with
`ucimlrepo` when it runs; no raw dataset file is currently stored here.

- Target value `1` means good credit risk.
- Target value `2` means bad credit risk.
- Generated exploratory plots are preserved in `data/eda/`.
- Raw data belongs in `data/raw/` and is ignored by Git.

The original baseline report and its figures remain available at
`docs/stage1_report.md` and `data/eda/`.
