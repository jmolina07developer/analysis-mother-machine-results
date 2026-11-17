# Analysis of mother machine results

Jupyter notebooks that take the Excel sheets of the mother cells written by the
[mother machine pipeline](https://github.com/jmolina07developer/mother-machine-pipeline)
(`<area>-<date>_longitud_madres.xlsx`) and run different analyses on them.

Every notebook works the same way:

1. Open the notebook and fill in the cell **Your data**: one line per folder of Excel files, with the areas
   you want and the condition they belong to (and, if you know it, the time the flow started).
2. **Run › Run All Cells**.
3. The figures (PNG, 200 dpi, sized for slides) and tables (CSV) are saved in `results/<notebook>/<RUN_NAME>/`.

```python
EXPERIMENTS = [
    dict(folder='../November/6/Stacks/Excel', areas='1-10',  condition='NIT+GEN', flow_start='10:17'),
    dict(folder='../November/6/Stacks/Excel', areas='11-20', condition='NIT',     flow_start='10:17'),
    dict(folder='../October/16/Stacks/Excel', areas='all',   condition='LB'),      # no flow_start: time 0 = first image
]
```

## Notebooks

| Notebook | What it does |
|---|---|
| `01_division_phenotypes.ipynb` | Each mother as the vector of its cumulative number of divisions $N(t)$ (e.g. `0001111222233334`); hierarchical clustering of all the mothers together; heatmaps ordered by the dendrogram; fraction of each phenotype in each condition with bootstrap intervals over areas |

## Installation

Python ≥ 3.10 with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
jupyter lab
```

If you already use the environment of the mother machine pipeline, it has everything that is needed.

## Layout

```
01_division_phenotypes.ipynb    the analyses (one notebook each)
mmtools/                        code shared by the notebooks
    excel.py                    reading the Excel files and the conditions
    trajectories.py             per-mother trajectories on a time grid
    clustering.py               clustering, choice of the number of clusters, fractions
    plots.py                    figures
results/                        outputs (not tracked by git)
```

## Conventions

- **Time 0** is the start of the flow when `flow_start` is given; otherwise the first image of the folder.
- A **division** is counted at the first image where it is seen (the image where the daughter's length is written).
- A mother is **lost** when its top cell changes without a division (a comment in the Excel) or the channel
  empties for good; lost mothers are left out of the analyses that need the whole trajectory.
- Uncertainties are bootstraps over **areas** (mothers of the same area are not independent).

## License

GPL-3.0, see `LICENSE`.
