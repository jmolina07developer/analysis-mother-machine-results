# Analysis of mother machine results

Jupyter notebooks that take the Excel sheets of the mother cells written by the
[mother machine pipeline](https://github.com/jmolina07developer/mother-machine-pipeline)
(`<area>-<date>_longitud_madres.xlsx`) and run different analyses on them.

There are two ways to use them: a **graphical interface** in the web browser, for people who do not program,
and the **notebooks**, which explain every step. Both use the same code (`mmtools/`) and give the same results.

## Graphical interface (no programming)

1. **Install once**: download this folder (**Code › Download ZIP**, then unzip it) and double-click
   `install_windows.bat` (Windows) or `install_mac.command` (macOS; the first time: right-click › Open › Open).
   It installs Miniforge (conda) if needed and an environment called `mmanalysis` (~5 min, needs internet).
2. **Open it**: double-click `start_windows.bat` (or the *Mother machine analyses* shortcut on the Desktop) or
   `start_mac.command`. The interface opens in the web browser (http://localhost:8502); keep the black window
   open while you use it, close it to stop.
3. **1 · Experiments**: choose each folder of Excel files and write the condition of each group of areas (and the
   flow start, if you know it). The page checks that everything can be read.
4. Choose an analysis on the left, adjust the settings (every one has a ⓘ) and press **▶ Run**. The figures and
   tables appear on the page and are saved in the results folder, in `<analysis>/<run name>/`, together with a
   `settings.json` that **↩ Load a previous run** reads back to repeat or change the run.

The experiments and the last settings of each analysis are remembered between sessions.

## Notebooks

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

## The analyses (interface page / notebook)

| Notebook | What it does |
|---|---|
| **2** · `01_division_phenotypes.ipynb` | Each mother as the vector of its cumulative number of divisions $N(t)$ (e.g. `0001111222233334`); hierarchical clustering of all the mothers together; heatmaps ordered by the dendrogram; fraction of each phenotype in each condition with bootstrap intervals over areas |
| **3** · `02_length_phenotypes.ipynb` | The same on the length of the mother $L(t)$ (saw-tooth = dividing, flat = arrested, brighter and brighter = filament); options for the mothers past the channel exit and µm/px; lengths drawn in the order of the division clustering and cross-table of the two descriptions |
| **4** · `03_growth_rate_length_bliss.ipynb` | The elongation velocity $v = \lambda L$ split exactly into rate × length × coupling; given a control, two monotherapies and their combination, the log-effect of each drug on each part and the Bliss prediction of the combination for a chosen window of time (95 % bootstrap over areas), and the same along time |
| **5** · `04_quantile_bliss.ipynb` | Bliss on the whole distributions (quantile functions) of $v$, $\lambda$ and $L$: Bliss at every rank, the effects as vectors in 3-d (each drug, their sum, Bliss, observed), and the plane $Q_{AB}\approx Q_{ctrl}\rho_A^{\alpha}\rho_B^{\beta}$ with a bootstrap cloud of $(\alpha,\beta)$ (Bliss = (1,1), B alone = (0,1)); sensitivity to the window |

## Installation without the installers

Python 3.10–3.12 with the packages in `requirements.txt` (tested versions):

```bash
conda env create -f environment.yml && conda activate mmanalysis     # or: pip install -r requirements.txt
python check_install.py
streamlit run app.py          # the interface
jupyter lab                   # the notebooks
```

## Using the analyses from a script

Each notebook is also one function that returns its figures and tables without printing anything, so a
graphical interface only has to collect the experiments and the settings and show the result:

```python
import mmtools as mm

problems = mm.check_entries(EXPERIMENTS)          # list of readable messages; [] = everything can be read
res = mm.run_length_phenotypes(EXPERIMENTS, step=10, t_end=170, past_exit='exclude',
                               progress=lambda done, total: print(done, '/', total))
res.figures      # {name: matplotlib Figure}, in order
res.tables       # {name: pandas DataFrame}
res.messages     # warnings for the user
res.save('results/02_length_phenotypes/my_run')   # PNG + CSV + settings.json

mm.ANALYSES                                       # {'01_division_phenotypes': run_division_phenotypes, ...}
mm.entries_table(EXPERIMENTS)                     # the experiments as a table (e.g. for an editable grid)
experiments, settings = mm.load_config('results/.../settings.json')   # re-run a saved analysis
```

## Layout

```
app.py                          the graphical interface (streamlit)
install_*/start_*               installers and launchers (Windows / macOS)
01_division_phenotypes.ipynb    the analyses (one notebook each)
02_length_phenotypes.ipynb
03_growth_rate_length_bliss.ipynb
04_quantile_bliss.ipynb
mmtools/                        code shared by the notebooks
    config.py                   the experiments (folder, areas, condition, time 0): checking, saving, loading
    excel.py                    reading the Excel files
    trajectories.py             per-mother trajectories on a time grid
    clustering.py               clustering, choice of the number of clusters, fractions
    plots.py                    figures
    growth.py                   rate × length split, log-effects, Bliss
    quantile_bliss.py           Bliss on quantile functions, (α, β) plane, 3-d vectors
    analyses.py                 each analysis as functions (steps for the notebooks, one call for an interface)
results/                        outputs (not tracked by git)
```

## Conventions

- **Time 0** is the start of the flow when `flow_start` is given; otherwise the first image of the folder.
- A grid point $t$ takes the last image at or before $t$ (lengths: if at most 12 min older).
- A **division** is counted at the first image where it is seen (the image where the daughter's length is written).
- A mother is **lost** when its top cell changes without a division (a comment in the Excel) or the channel
  empties for good; lost mothers are left out of the analyses that need the whole trajectory.
- Uncertainties are bootstraps over **areas** (mothers of the same area are not independent).

## License

GPL-3.0, see `LICENSE`.
