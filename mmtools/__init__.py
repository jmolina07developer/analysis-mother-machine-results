"""
Helpers shared by the analysis notebooks: reading the mother Excel files, trajectories, clustering, figures.

For a script or a graphical interface, each analysis is one call that returns its figures and tables:

    import mmtools as mm
    problems = mm.check_entries(EXPERIMENTS)            # [] if every folder / area / time can be read
    res = mm.run_length_phenotypes(EXPERIMENTS, step=10, t_end=170, progress=lambda done, total: ...)
    res.figures, res.tables, res.messages
    res.save('results/02_length_phenotypes/my_run')

mm.ANALYSES lists them by name.
"""
from .config import Entry, as_entries, entries_table, check_entries, save_config, load_config
from .excel import load_experiments, read_mother_excel, parse_areas
from .trajectories import time_grid, default_t_end, division_vectors, length_vectors, fill_gaps, as_word
from .clustering import cluster, choose_k, numbered_clusters, fractions
from .analyses import (Dataset, Phenotypes, AnalysisResult, load_dataset, division_matrix, length_matrix,
                       find_phenotypes, division_phenotypes, length_phenotypes, lengths_in_division_order,
                       compare_phenotypes, run_division_phenotypes, run_length_phenotypes,
                       GrowthSplit, growth_split, run_growth_bliss, QuantileBlissSet, quantile_bliss_analysis,
                       quantile_bliss_windows, run_quantile_bliss, ANALYSES)
from . import growth, quantile_bliss
