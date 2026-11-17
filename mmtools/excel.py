"""
Read the mother-cell Excel sheets written by the mother machine pipeline (excel_madres.py).

One file per area:  <area>-<date>_longitud_madres.xlsx
    row 1: time of each image (HH:MM:SS)
    row 2: image number
    then two rows per channel:
      upper: length (px) of the mother (the cell at the closed end of the channel) in each image
      lower: only in the first image where a division of the mother is seen, the length of the daughter
    column A: channel number
Cell colours / comments used here:
    red NaN   the mother is past the channel exit (no length)
    comment   the top cell changed without a division: the mother was lost by the tracker
    empty     no cell in the channel
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

RED = 'FFFF7B7B'
FILE_RE = re.compile(r'^(\d+)-(\d+)_longitud_madres\.xlsx$')


# ------------------------------------------------------------------ small parsers
def parse_areas(spec):
    """'all' / None -> None (every area);  '1-10' / '1-5, 8, 12-14' / [1, 2, 3] -> set of ints"""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ('', 'all')):
        return None
    if isinstance(spec, (int, np.integer)):
        return {int(spec)}
    if not isinstance(spec, str):
        return {int(a) for a in spec}
    out = set()
    for part in spec.replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def clock_to_min(s):
    """'HH:MM' or 'HH:MM:SS' -> minutes since midnight"""
    parts = [int(float(x)) for x in str(s).strip().split(':')]
    parts += [0] * (3 - len(parts))
    return parts[0] * 60 + parts[1] + parts[2] / 60


def _times(header):
    """
    Time of each image from row 1. Returns (minutes, is_clock).
    Clock times (HH:MM:SS) are minutes since midnight (a run past midnight keeps increasing);
    '12.3 min' (written when the pipeline had no clock) are minutes since the first image.
    """
    vals = [str(v).strip() for v in header]
    if all(':' in v for v in vals):
        t = np.array([clock_to_min(v) for v in vals])
        t = t + 1440 * np.r_[0, np.cumsum(np.diff(t) < -720)]           # past midnight
        return t, True
    if all(v.endswith('min') for v in vals):
        return np.array([float(v.replace('min', '')) for v in vals]), False
    raise ValueError(f'cannot read the image times in row 1: {vals[:3]} ...')


def _num(v):
    if v is None or isinstance(v, str):
        return np.nan
    return float(v)


# ------------------------------------------------------------------ one Excel file
def read_mother_excel(path):
    """
    One row per channel and image:
        channel, image, clock (min), L (px), L_daughter (px, 0 if no division seen),
        divides, lost_here (top cell changed without a division), empty, past_exit
    """
    ws = openpyxl.load_workbook(path).worksheets[0]
    rows = list(ws.iter_rows())
    t, is_clock = _times([c.value for c in rows[0][1:]])
    images = [c.value for c in rows[1][1:]]
    out = []
    for i in range(2, len(rows), 2):
        mom = rows[i][1:]
        dau = rows[i + 1][1:] if i + 1 < len(rows) else [None] * len(mom)
        ch = rows[i][0].value
        if ch is None:
            continue
        for j, cm in enumerate(mom[:len(t)]):
            cd = dau[j] if j < len(dau) else None
            vd = getattr(cd, 'value', None)
            out.append(dict(
                channel=int(ch), image=images[j], clock=t[j],
                L=_num(cm.value), L_daughter=0.0 if vd is None else _num(vd),
                comment=cm.comment is not None or getattr(cd, 'comment', None) is not None,
                empty=cm.value is None,
                past_exit=bool(cm.fill is not None and cm.fill.fill_type and cm.fill.fgColor.rgb == RED)))
    d = pd.DataFrame(out)
    d['divides'] = d.L_daughter.fillna(1.0) != 0                         # a daughter length (or a NaN daughter) = a division
    d['lost_here'] = d.comment & ~d.divides
    d.attrs['is_clock'] = is_clock
    return d


def list_excels(folder):
    """{area (int): path} of the *_longitud_madres.xlsx files of a folder"""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'folder not found: {folder.resolve()}')
    out = {}
    for f in sorted(folder.iterdir()):
        m = FILE_RE.match(f.name)
        if m and not f.name.startswith('~$'):
            out[int(m.group(1))] = f
    if not out:
        raise FileNotFoundError(f'no *_longitud_madres.xlsx files in {folder.resolve()}')
    return out


# ------------------------------------------------------------------ all the experiments
def load_experiments(experiments, verbose=True):
    """
    experiments: list of dicts, one per (folder, condition):
        folder      folder with the <area>-<date>_longitud_madres.xlsx files
        condition   name of the condition (several entries can share one condition)
        areas       'all' (default), '1-10', '1-5, 8', [1, 2, 3]
        flow_start  'HH:MM' start of the flow; None -> the first image of the folder is time 0
        name        optional label of the experiment (default: the date in the file names)

    Returns one long table (one row per mother and image) with
        condition, experiment, area, channel, mother (unique id), image, t (min since time 0),
        L, L_daughter, divides, lost (the mother has already been lost at this image), empty, past_exit
    and a summary table (one row per entry).
    """
    frames, summary, cache = [], [], {}
    for k, e in enumerate(experiments):
        folder = Path(e['folder'])
        files = list_excels(folder)
        if folder not in cache:                                          # read each folder once
            cache[folder] = {a: read_mother_excel(p) for a, p in files.items()}
        sheets = cache[folder]
        want = parse_areas(e.get('areas', 'all'))
        areas = sorted(sheets) if want is None else sorted(want)
        missing = [a for a in areas if a not in sheets]
        if missing:
            raise FileNotFoundError(f"entry {k} ({e['condition']}): no Excel file for areas {missing} in {folder}")
        is_clock = all(s.attrs['is_clock'] for s in sheets.values())
        if e.get('flow_start') and is_clock:
            t0, t0_from = clock_to_min(e['flow_start']), f"flow start {e['flow_start']}"
        else:
            t0 = min(s.clock.min() for s in sheets.values())               # first image of the whole folder
            t0_from = 'first image of the folder'
            if e.get('flow_start') and not is_clock:
                t0_from += ' (the Excel has no clock times, flow_start ignored)'
        date = FILE_RE.match(files[areas[0]].name).group(2)
        name = e.get('name') or (f'{date[:2]}/{date[2:4]}/{date[4:]}' if len(date) == 8 else date)
        for a in areas:
            d = sheets[a].copy()
            d['t'] = d.clock - t0
            d['condition'], d['experiment'], d['area'] = e['condition'], name, a
            frames.append(d)
        summary.append(dict(condition=e['condition'], experiment=name, folder=str(folder), areas=f'{areas[0]}–{areas[-1]}' if areas == list(range(areas[0], areas[-1] + 1)) else ', '.join(map(str, areas)),
                            n_areas=len(areas), time_zero=t0_from,
                            last_image_min=round(max(sheets[a].clock.max() - t0 for a in areas), 1)))
    LM = pd.concat(frames, ignore_index=True)
    LM = LM.sort_values(['condition', 'experiment', 'area', 'channel', 'image']).reset_index(drop=True)
    LM['mother'] = LM.groupby(['condition', 'experiment', 'area', 'channel'], sort=False).ngroup()
    # a mother is lost from the first image where its top cell changes without a division
    LM['lost'] = LM.groupby('mother').lost_here.cummax()
    # ... or from the image where the channel becomes empty for good
    gone = LM.iloc[::-1].groupby('mother')['empty'].cummin().iloc[::-1].astype(bool)
    LM['lost'] |= gone
    cols = ['condition', 'experiment', 'area', 'channel', 'mother', 'image', 't', 'L', 'L_daughter',
            'divides', 'lost', 'empty', 'past_exit']
    S = pd.DataFrame(summary)
    if verbose:
        print(f"{LM.mother.nunique()} channels in {S.n_areas.sum()} areas, {len(S)} entries")
    return LM[cols], S
