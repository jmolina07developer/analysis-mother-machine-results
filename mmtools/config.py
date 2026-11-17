"""
The experiments of an analysis: which folders, which areas, which condition, when time 0 is.

The same description is used by the notebooks (a list of dicts in the first cell) and can be used by a
graphical interface (an editable table, or a JSON file saved and loaded with `save_config` / `load_config`).
"""
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

CLOCK_RE = re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*$')
AREAS_RE = re.compile(r'^\s*(all|\d+(\s*-\s*\d+)?(\s*[,;]\s*\d+(\s*-\s*\d+)?)*)\s*$', re.I)


@dataclass
class Entry:
    """One folder of Excel files (or some of its areas) and the condition they belong to."""
    folder: str
    condition: str
    areas: str = 'all'            # 'all', '1-10', '1-5, 8, 12-14'
    flow_start: str | None = None  # 'HH:MM'; None -> time 0 = first image of the folder
    name: str | None = None        # label of the experiment; None -> the date in the file names

    def as_dict(self):
        return asdict(self)


def as_entries(experiments):
    """list of dicts / Entry objects, or a DataFrame with those columns -> list of Entry"""
    if isinstance(experiments, pd.DataFrame):
        experiments = experiments.to_dict('records')
    out = []
    for e in experiments:
        if isinstance(e, Entry):
            out.append(e); continue
        e = {k: (None if (isinstance(v, float) and v != v) or v == '' else v) for k, v in dict(e).items()}
        areas = e.get('areas', 'all')
        if areas is None:
            areas = 'all'
        elif not isinstance(areas, str):
            areas = ', '.join(str(int(a)) for a in (areas if hasattr(areas, '__iter__') else [areas]))
        out.append(Entry(folder=str(e['folder']), condition=str(e['condition']), areas=areas,
                         flow_start=e.get('flow_start'), name=e.get('name')))
    return out


def entries_table(experiments):
    """list of Entry -> DataFrame (one row per entry), e.g. for an editable table in an interface"""
    return pd.DataFrame([e.as_dict() for e in as_entries(experiments)],
                        columns=['folder', 'condition', 'areas', 'flow_start', 'name'])


def check_entries(experiments, base=None):
    """
    List of problems (empty list = everything is fine): missing folders, folders without Excel files,
    areas without a file, badly written areas or flow start times, empty conditions.
    base: folder relative paths are taken from (default: the current folder).
    """
    from .excel import list_excels, parse_areas
    problems, used = [], {}
    try:
        entries = as_entries(experiments)
    except Exception as err:                                     # e.g. a missing key
        return [f'cannot read the experiments: {err}']
    if not entries:
        return ['no experiments given']
    for i, e in enumerate(entries, 1):
        where = f'line {i} ({e.condition or "no condition"})'
        if not e.condition.strip():
            problems.append(f'{where}: the condition is empty')
        folder = Path(base or '.') / e.folder
        if not folder.is_dir():
            problems.append(f'{where}: folder not found: {folder}')
            continue
        try:
            files = list_excels(folder)
        except FileNotFoundError as err:
            problems.append(f'{where}: {err}'); continue
        if not AREAS_RE.match(str(e.areas)):
            problems.append(f"{where}: cannot read the areas '{e.areas}' (write 'all', '1-10' or '1-5, 8')")
        else:
            want = parse_areas(e.areas)
            missing = sorted(want - set(files)) if want else []
            if missing:
                problems.append(f'{where}: no Excel file for areas {missing} (there are {min(files)}–{max(files)})')
            key = folder.resolve()
            for a in sorted((want or set(files)) & set(files)):
                if (key, a) in used:
                    problems.append(f'{where}: area {a} of this folder is already in line {used[(key, a)]}'
                                    ' (each area can be in only one line)')
                    break
                used[(key, a)] = i
        if e.flow_start and not CLOCK_RE.match(str(e.flow_start)):
            problems.append(f"{where}: cannot read the flow start '{e.flow_start}' (write HH:MM)")
    return problems


def save_config(path, experiments, **settings):
    """save the experiments and the settings of an analysis to a JSON file"""
    data = dict(experiments=[e.as_dict() for e in as_entries(experiments)], settings=settings)
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def load_config(path):
    """-> (list of Entry, dict of settings)"""
    data = json.loads(Path(path).read_text())
    return as_entries(data['experiments']), data.get('settings', {})
