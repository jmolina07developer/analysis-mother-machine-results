"""
Graphical interface of the analyses (runs in the web browser, on your own computer).

Start it with the launcher (start_mac.command / start_windows.bat) or, from this folder, with
    streamlit run app.py

It does exactly what the notebooks do (it calls the same functions of mmtools) and writes the same files:
    <results folder>/<analysis>/<run name>/   figures (PNG), tables (CSV), settings.json, messages.txt
"""
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mmtools as mm  # noqa: E402
from mmtools.excel import list_excels, FILE_RE  # noqa: E402

st.set_page_config(page_title='Mother machine · analyses', page_icon='📊', layout='wide')
MEMORY = Path.home() / '.mother_machine_analyses.json'
COLUMNS = ['folder', 'condition', 'areas', 'flow_start', 'name']
ANALYSES = {
    'divisions': dict(page='2 · Division phenotypes', key='01_division_phenotypes', run=mm.run_division_phenotypes),
    'lengths': dict(page='3 · Length phenotypes', key='02_length_phenotypes', run=mm.run_length_phenotypes),
    'growth': dict(page='4 · Rate × length and Bliss', key='03_growth_rate_length_bliss', run=mm.run_growth_bliss),
    'quantiles': dict(page='5 · Bliss on quantiles', key='04_quantile_bliss', run=mm.run_quantile_bliss),
}
PAGES = ['1 · Experiments'] + [a['page'] for a in ANALYSES.values()] + ['Help']


# ============================================================================================== helpers
def read_json(f, default=None):
    try:
        return json.loads(Path(f).read_text())
    except Exception:
        return {} if default is None else default


def write_json(f, d):
    try:
        Path(f).write_text(json.dumps(d, indent=1, default=str))
    except Exception:
        pass


def remember(**kw):
    d = read_json(MEMORY, {})
    d.update(kw)
    write_json(MEMORY, d)


def open_path(path):
    """open a file or folder with the default application of the computer"""
    path = str(path)
    try:
        if sys.platform.startswith('win'):
            os.startfile(path)  # noqa: S606
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception as e:
        st.error(f'Could not open {path}: {e}')


def dialog(kind, title):
    """folder / file dialog of the operating system, in a separate process so it does not block the interface"""
    if kind == 'folder':
        call = f'filedialog.askdirectory(title={title!r})'
    else:
        call = f'filedialog.askopenfilename(title={title!r}, filetypes=[("settings", "*.json"), ("all", "*.*")])'
    code = ('import tkinter as tk; from tkinter import filedialog; r = tk.Tk(); r.withdraw(); '
            f'r.attributes("-topmost", True); print({call})')
    try:
        out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=900)
        return out.stdout.strip() or None
    except Exception:
        return None


def clean_table(df):
    """experiments table with the five text columns, empty rows removed"""
    df = pd.DataFrame(df).reindex(columns=COLUMNS)
    df = df.astype(object).where(df.notna(), '')
    df = df.map(lambda x: str(x).strip())
    return df[(df.folder != '') | (df.condition != '')].reset_index(drop=True)


def entries_of(df):
    out = []
    for r in clean_table(df).to_dict('records'):
        out.append(dict(folder=r['folder'], condition=r['condition'], areas=r['areas'] or 'all',
                        flow_start=r['flow_start'] or None, name=r['name'] or None))
    return out


def safe_name(s):
    s = re.sub(r'[^\w\-+.]+', '_', str(s).strip()).strip('_')
    return s or datetime.now().strftime('run_%Y%m%d_%H%M')


def zip_folder(folder):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(Path(folder).iterdir()):
            z.write(p, p.name)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def folder_info(folder, stamp):
    """areas, date and image times of a folder of Excel files"""
    import openpyxl
    files = list_excels(folder)
    first = files[min(files)]
    ws = openpyxl.load_workbook(first, read_only=True).worksheets[0]
    times = [c for c in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))[1:] if c]
    date = FILE_RE.match(first.name).group(2)
    return dict(areas=sorted(files), date=f'{date[:2]}/{date[2:4]}/{date[4:]}' if len(date) == 8 else date,
                first=str(times[0]) if times else '?', last=str(times[-1]) if times else '?', n_images=len(times))


def stamp_of(folders):
    """latest modification time of the Excel files of some folders (to know when to read them again)"""
    t = 0.0
    for f in folders:
        try:
            t = max([t] + [p.stat().st_mtime for p in Path(f).glob('*_longitud_madres.xlsx')])
        except Exception:
            pass
    return t


@st.cache_data(show_spinner='Reading the Excel files…')
def inspect(entries_json, stamp):
    LM, S = mm.load_experiments(json.loads(entries_json))
    mothers = LM.groupby('condition').mother.nunique()
    return S, mothers


# ============================================================================================== state
mem = read_json(MEMORY, {})
if 'exp_base' not in st.session_state:
    st.session_state.exp_base = clean_table(pd.DataFrame(mem.get('experiments', []), columns=COLUMNS))
    st.session_state.exp_version = 0
    st.session_state.results_dir = mem.get('results_dir') or str(HERE / 'results')
    st.session_state.settings = mem.get('settings', {})
    st.session_state.runs = {}
    st.session_state.new_folder = ''


def set_table(df):
    st.session_state.exp_base = clean_table(df)
    st.session_state.exp_current = st.session_state.exp_base
    st.session_state.exp_version += 1
    remember(experiments=st.session_state.exp_base.to_dict('records'))


# ============================================================================================== sidebar
st.sidebar.title('📊 Mother machine · analyses')
page = st.sidebar.radio('Go to', PAGES, key='page')
st.sidebar.divider()
st.sidebar.subheader('Results folder')
if st.sidebar.button('📂 Choose the results folder…', width='stretch'):
    r = dialog('folder', 'Folder where the results are saved')
    if r:
        st.session_state.results_dir = r
        st.session_state.pop('results_dir_field', None)
rd = st.sidebar.text_input('Results are saved in', value=st.session_state.results_dir, key='results_dir_field',
                           help='Each run is saved in <this folder>/<analysis>/<run name>/')
if rd != st.session_state.results_dir:
    st.session_state.results_dir = rd
remember(results_dir=st.session_state.results_dir)
RESULTS = Path(st.session_state.results_dir).expanduser()
if st.sidebar.button('Open the results folder', disabled=not RESULTS.exists(), width='stretch'):
    open_path(RESULTS)
st.sidebar.divider()
if st.sidebar.button('↩ Load a previous run (settings.json)…', width='stretch',
                     help='Every run saves a settings.json with its experiments and settings: load it to repeat or '
                          'change that run.'):
    f = dialog('file', 'settings.json of a previous run')
    if f:
        try:
            exps, sets = mm.load_config(f)
            set_table(mm.entries_table(exps))
            an = sets.pop('analysis', None)
            if an:
                key = next((k for k, a in ANALYSES.items() if a['key'] == an), None)
                if key:
                    st.session_state.settings[key] = sets
                    remember(settings=st.session_state.settings)
            st.sidebar.success(f'Loaded: {len(exps)} lines' + (f' and the settings of {an}' if an else ''))
        except Exception as e:
            st.sidebar.error(f'Could not read {f}: {e}')


# ============================================================================================== 1. experiments
def page_experiments():
    if st.session_state.get('last_page') != PAGES[0] and 'exp_current' in st.session_state:
        # coming back from another page: show the table as it was left
        if not st.session_state.exp_current.equals(st.session_state.exp_base):
            st.session_state.exp_base = st.session_state.exp_current
            st.session_state.exp_version += 1
    st.title('1 · Experiments')
    st.markdown(
        'Tell the program **where the Excel files are** (`<area>-<date>_longitud_madres.xlsx`, written by the mother '
        'machine pipeline in `Stacks/Excel`) and **which condition each group of areas is**. One line per folder and '
        'condition: if one folder has several conditions (e.g. areas 1–10 one drug, 11–20 another), add one line for '
        'each. Several lines can have the same condition (e.g. two experiments of the same treatment).')

    st.subheader('Add a folder')
    c1, c2 = st.columns([1, 3])
    if c1.button('📂 Choose a folder of Excel files…', width='stretch'):
        r = dialog('folder', 'Folder with the *_longitud_madres.xlsx files')
        if r:
            st.session_state.new_folder = r
    folder = c2.text_input('Folder', value=st.session_state.new_folder, label_visibility='collapsed',
                           placeholder='…or paste the path of the folder here')
    st.session_state.new_folder = folder
    if folder:
        try:
            info = folder_info(folder, stamp_of([folder]))
            a = info['areas']
            st.success(f"**{len(a)} Excel files**, areas {a[0]}–{a[-1]}, date {info['date']}; "
                       f"{info['n_images']} images from {info['first']} to {info['last']} (area {a[0]}).")
            with st.form('add_line', clear_on_submit=False, border=True):
                f1, f2, f3, f4 = st.columns([2, 2, 1.3, 1.3])
                cond = f1.text_input('Condition', placeholder='e.g. NIT+GEN',
                                     help='Name of the condition. Lines with the same name are pooled.')
                areas = f2.text_input('Areas', value='all', help="'all', a range '1-10', or a list '1-5, 8, 12-14'")
                flow = f3.text_input('Flow start (HH:MM)', placeholder='optional',
                                     help='Time the flow started: it becomes time 0. Empty: time 0 is the first image '
                                          'of this folder.')
                name = f4.text_input('Experiment name', placeholder=info['date'],
                                     help='Optional label of the experiment (default: the date in the file names).')
                if st.form_submit_button('➕ Add this line', type='primary'):
                    if not cond.strip():
                        st.error('Write the condition first.')
                    else:
                        row = dict(folder=folder, condition=cond, areas=areas or 'all', flow_start=flow, name=name)
                        set_table(pd.concat([current_table(), pd.DataFrame([row])], ignore_index=True))
                        st.rerun()
            st.caption('You can add several lines from the same folder (one per condition) before choosing another one.')
        except FileNotFoundError as e:
            st.error(str(e))

    st.subheader('Lines')
    st.caption('Edit any cell directly. To delete a line, select it (left edge) and press Delete, or use the bin icon.')
    edited = st.data_editor(
        st.session_state.exp_base, key=f'editor_{st.session_state.exp_version}', num_rows='dynamic', width='stretch',
        hide_index=True,
        column_config={
            'folder': st.column_config.TextColumn('folder', width='large', help='Folder with the Excel files'),
            'condition': st.column_config.TextColumn('condition', help='Name of the condition'),
            'areas': st.column_config.TextColumn('areas', help="'all', '1-10' or '1-5, 8'"),
            'flow_start': st.column_config.TextColumn('flow start', help='HH:MM; empty = first image of the folder'),
            'name': st.column_config.TextColumn('experiment name', help='Optional'),
        })
    table = clean_table(edited)
    st.session_state.exp_current = table
    remember(experiments=table.to_dict('records'))
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button('🗑 Remove all lines', disabled=table.empty):
        set_table(pd.DataFrame(columns=COLUMNS)); st.rerun()
    if c2.button('🔄 Read the files again', help='If you have changed the Excel files'):
        inspect.clear(); folder_info.clear(); st.rerun()

    if table.empty:
        st.info('No lines yet: choose a folder above.')
        return
    st.subheader('Check')
    entries = entries_of(table)
    problems = mm.check_entries(entries)
    if problems:
        for p in problems:
            st.error(p)
        return
    S, mothers = inspect(json.dumps(entries), stamp_of({e['folder'] for e in entries}))
    conds = list(dict.fromkeys(e['condition'] for e in entries))
    st.success(f"Everything can be read: **{len(conds)} conditions** ({', '.join(conds)}), "
               f"{int(S.n_areas.sum())} areas, {int(mothers.sum())} channels. "
               f"Every condition has images up to **{S.last_image_min.min():.0f} min**.")
    show = S.rename(columns={'n_areas': 'areas (n)', 'time_zero': 'time 0', 'last_image_min': 'last image (min)'})
    st.dataframe(show.drop(columns='folder'), hide_index=True, width='stretch')
    st.markdown('Now choose an analysis in the panel on the left.')


def current_table():
    """the experiments table as last edited on page 1"""
    return clean_table(st.session_state.get('exp_current', st.session_state.exp_base))


# ============================================================================================== analyses
def need_data():
    """experiments of page 1, checked; stops the page if they are not ready"""
    table = current_table()
    entries = entries_of(table)
    if not entries:
        st.warning('First add the experiments in **1 · Experiments**.'); st.stop()
    problems = mm.check_entries(entries)
    if problems:
        st.error('The experiments have problems (see **1 · Experiments**):\n\n- ' + '\n- '.join(problems)); st.stop()
    S, _ = inspect(json.dumps(entries), stamp_of({e['folder'] for e in entries}))
    conds = list(dict.fromkeys(e['condition'] for e in entries))
    return entries, conds, float(S.last_image_min.min())


def get(key, name, default):
    return st.session_state.settings.get(key, {}).get(name, default)


def t_end_input(key, t_max, help_extra=''):
    v = st.number_input('Use the data up to (min since time 0)', min_value=0.0, max_value=float(np.ceil(t_max)),
                        value=float(min(get(key, 't_end', 0) or 0, np.ceil(t_max))), step=5.0,
                        help='Nothing after this time is used (e.g. a gap in the imaging). 0 = up to the latest time '
                             f'that every condition has images for ({t_max:.0f} min).' + help_extra)
    return None if v == 0 else v


def roles_input(key, conds):
    st.markdown('**Which condition is which**')
    c = st.columns(4)
    labels = [('control', 'Control (no drug)'), ('A', 'Drug A'), ('B', 'Drug B'), ('combination', 'Combination A + B')]
    saved = get(key, 'roles', {}) or {}
    roles = {}
    for i, (r, lab) in enumerate(labels):
        default = saved.get(r) if saved.get(r) in conds else conds[min(i, len(conds) - 1)]
        roles[r] = c[i].selectbox(lab, conds, index=conds.index(default), key=f'{key}_role_{r}')
    if len(set(roles.values())) < 4:
        st.warning('The four roles must be four different conditions.')
    return roles


def window_input(key, t_max, label='Window of the analysis (min since time 0)'):
    hi = float(np.floor(t_max))
    w = get(key, 'window', None) or [max(0.0, hi - 45), hi]
    w = [min(max(0.0, float(w[0])), hi), min(float(w[1]), hi)]
    if w[0] >= w[1]: w = [max(0.0, hi - 45), hi]
    return st.slider(label, 0.0, hi, (w[0], w[1]), step=5.0, format='%.0f',
                     help='The per-mother velocity, rate and length are averaged over this window. Use the time-course '
                          'figure of 4 · Rate × length to choose it: where the effects have settled.')


def run_box(key, title, runner, kwargs, settings):
    """run name, Run button, progress; saves the result and remembers the settings"""
    default = get(key, 'run_name', '') or datetime.now().strftime('run_%Y%m%d_%H%M')
    c1, c2 = st.columns([3, 1])
    name = c1.text_input('Name of this run (folder of the results)', value=default, key=f'{key}_run_name')
    c2.write(''); c2.write('')
    go = c2.button('▶ Run', type='primary', width='stretch', key=f'{key}_run')
    if not go:
        return
    st.session_state.settings[key] = {**settings, 'run_name': name}
    remember(settings=st.session_state.settings)
    bar = st.progress(0.0, text='Reading the Excel files…')
    prog = lambda d, t: bar.progress(min(d / max(t, 1), 1.0), text=f'Bootstrap… {d}/{t}')
    try:
        res = runner(progress=prog, **kwargs)
    except ValueError as e:
        bar.empty(); st.error(str(e)); return
    except Exception as e:  # noqa: BLE001
        bar.empty(); st.error('Something went wrong:'); st.exception(e); return
    bar.progress(1.0, text='Saving…')
    out = RESULTS / ANALYSES[key]['key'] / safe_name(name)
    if out.exists() and any(out.iterdir()):
        k = 2
        while (out.parent / f'{out.name}_{k}').exists(): k += 1
        out = out.parent / f'{out.name}_{k}'
    res.save(out, close=True)
    bar.empty()
    st.session_state.runs[key] = dict(folder=str(out), res=res)


def show_run(key, tables):
    run = st.session_state.runs.get(key)
    if not run:
        return
    folder, res = Path(run['folder']), run['res']
    st.divider()
    st.subheader('Results')
    st.success(f'Saved in `{folder}`')
    c1, c2, _ = st.columns([1, 1, 2])
    if c1.button('📂 Open the folder', key=f'{key}_open'):
        open_path(folder)
    c2.download_button('⬇ Download all (zip)', data=zip_folder(folder), file_name=f'{folder.parent.name}_{folder.name}.zip',
                       key=f'{key}_zip')
    for m in res.messages:
        (st.warning if m.startswith('WARNING') else st.caption)(m)
    for title, t in tables(res):
        st.markdown(f'**{title}**')
        st.dataframe(t, width='stretch')
    for png in sorted(folder.glob('*.png')):
        st.image(str(png), caption=png.stem, width='stretch')


def fmt_fractions(res, key):
    P = res.objects.get(key)
    return P.fraction_table() if P is not None else res.tables['fractions']


def page_divisions():
    key = 'divisions'
    st.title('2 · Division phenotypes')
    st.markdown('Each mother as its **cumulative number of divisions** $N(t)$ on a time grid (e.g. `0001111222233`); '
                'all the mothers clustered together; heatmaps ordered by the dendrogram; **fraction of each phenotype '
                'in each condition**. Same as notebook `01_division_phenotypes.ipynb`.')
    entries, conds, t_max = need_data()
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1: t_end = t_end_input(key, t_max)
        step = c2.number_input('Time step of the vector (min)', 1.0, 60.0, float(get(key, 'step', 10)), 1.0,
                               help='One entry of the vector every this many minutes')
        opts = ['auto'] + list(range(2, 11))
        k = c3.selectbox('Number of phenotypes', opts, index=opts.index(get(key, 'n_phenotypes', 'auto')),
                         help="'auto' = the one with the highest silhouette")
        n_boot = st.select_slider('Resamples for the stability of the clusters', [0, 50, 100, 200, 500],
                                  value=get(key, 'n_boot', 100), help='0 = skip (faster)')
        sets = dict(t_end=t_end, step=step, n_phenotypes=k, n_boot=n_boot)
        run_box(key, 'divisions', mm.run_division_phenotypes,
                dict(experiments=entries, step=step, t_end=t_end, n_phenotypes=k, n_boot=n_boot), sets)
    show_run(key, lambda r: [('Mothers used', r.tables['mothers_used']),
                             ('Fraction of each phenotype in each condition, % (95 % interval over areas)',
                              fmt_fractions(r, 'divisions')),
                             ('Choice of the number of phenotypes', r.tables['k_selection'].round(3))])


def page_lengths():
    key = 'lengths'
    st.title('3 · Length phenotypes')
    st.markdown('Each mother as the vector of its **length** $L(t)$: saw-tooth = dividing, flat = arrested, brighter '
                'and brighter = filament. Clustering, heatmaps, fractions, and the comparison with the division '
                'phenotypes. Same as notebook `02_length_phenotypes.ipynb`.')
    entries, conds, t_max = need_data()
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1: t_end = t_end_input(key, t_max)
        step = c2.number_input('Time step of the vector (min)', 1.0, 60.0, float(get(key, 'step', 10)), 1.0)
        opts = ['auto'] + list(range(2, 11))
        k = c3.selectbox('Number of phenotypes', opts, index=opts.index(get(key, 'n_phenotypes', 'auto')),
                         help="'auto' = the one with the highest silhouette")
        c1, c2, c3, c4 = st.columns(4)
        pe = c1.radio('Mothers past the channel exit', ['exclude', 'cap'],
                      index=['exclude', 'cap'].index(get(key, 'past_exit', 'exclude')),
                      help="exclude: left out (they have no length); cap: given the channel length while they are out")
        ch = c2.number_input('Channel length (px), for "cap"', 0.0, 5000.0, float(get(key, 'channel_px', 0) or 0), 10.0,
                             help='0 = the longest length measured', disabled=pe != 'cap')
        um = c3.number_input('µm per pixel', 0.0, 10.0, float(get(key, 'um_per_px', 0) or 0), 0.001, format='%.4f',
                             help='0 = keep pixels')
        gap = c4.number_input('Fill gaps of up to (grid points)', 0, 10, int(get(key, 'max_gap', 2)), 1)
        n_boot = st.select_slider('Resamples for the stability of the clusters', [0, 50, 100, 200, 500],
                                  value=get(key, 'n_boot', 100), help='0 = skip (faster)')
        sets = dict(t_end=t_end, step=step, n_phenotypes=k, past_exit=pe, channel_px=ch or None, um_per_px=um or None,
                    max_gap=gap, n_boot=n_boot)
        run_box(key, 'lengths', mm.run_length_phenotypes,
                dict(experiments=entries, step=step, t_end=t_end, n_phenotypes=k, n_boot=n_boot, max_gap=gap,
                     past_exit=pe, channel_px=ch or None, um_per_px=um or None), sets)
    show_run(key, lambda r: [('Mothers used', r.tables['mothers_used']),
                             ('Fraction of each phenotype in each condition, % (95 % interval over areas)',
                              fmt_fractions(r, 'length')),
                             ('Division phenotypes (rows) against length phenotypes (columns)', r.tables['divisions_vs_length'])])


def page_growth():
    key = 'growth'
    st.title('4 · Rate × length and Bliss')
    st.markdown('The elongation velocity of each mother split **exactly** into rate × length, $v = \\lambda L$. Given a '
                'control, two drugs and their combination: the **effect of each drug on each part** and the **Bliss '
                'prediction** of the combination, in a window of time. Same as notebook '
                '`03_growth_rate_length_bliss.ipynb`.')
    entries, conds, t_max = need_data()
    if len(conds) < 4:
        st.warning('This analysis needs four conditions: control, drug A, drug B and the combination.'); st.stop()
    with st.container(border=True):
        roles = roles_input(key, conds)
        c1, c2 = st.columns([1, 2])
        with c1: t_end = t_end_input(key, t_max)
        with c2: window = window_input(key, t_end or t_max)
        with st.expander('More settings'):
            c1, c2, c3 = st.columns(3)
            um = c1.number_input('µm per pixel', 0.0, 10.0, float(get(key, 'um_per_px', 0) or 0), 0.001, format='%.4f',
                                 help='0 = keep pixels')
            cover = c2.slider('A mother counts if its intervals cover at least', 0.1, 1.0, float(get(key, 'cover', 0.6)), 0.05,
                              help='fraction of the window')
            dtr = c3.slider('Intervals between images allowed (min)', 0.0, 60.0, tuple(float(x) for x in get(key, 'dt_range', [3, 25])), 1.0)
            c1, c2 = st.columns(2)
            n_boot = c1.select_slider('Bootstrap resamples', [200, 500, 1000, 2000], value=get(key, 'n_boot', 1000))
            along = c2.checkbox('Also along time (30-min windows)', value=get(key, 'along_time', True))
        sets = dict(roles=roles, window=list(window), t_end=t_end, um_per_px=um or None, cover=cover, dt_range=list(dtr),
                    n_boot=n_boot, along_time=along)
        run_box(key, 'growth', mm.run_growth_bliss,
                dict(experiments=entries, roles=roles, window=window, t_end=t_end, cover=cover, dt_range=tuple(dtr),
                     um_per_px=um or None, n_boot=n_boot, along_time=along), sets)
    show_run(key, lambda r: [('Log-effects relative to the control (95 % interval over areas)', r.tables['log_effects']),
                             ('Bliss prediction in the units of the data', r.tables['bliss_prediction'].round(4)),
                             ('Mothers in the window', r.tables['window_summary'].round(4))])


def page_quantiles():
    key = 'quantiles'
    st.title('5 · Bliss on quantiles')
    st.markdown('Bliss on the **whole distributions** of velocity, rate and length: Bliss at every rank, the effects as '
                '**vectors in 3-d**, and **how much of each drug is in the combination**, the plane '
                '$Q_{AB}\\approx Q_{ctrl}\\,\\rho_A^{\\alpha}\\rho_B^{\\beta}$ ((1,1) = Bliss, (0,1) = B alone, '
                '(1,0) = A alone). Same as notebook `04_quantile_bliss.ipynb`.')
    entries, conds, t_max = need_data()
    if len(conds) < 4:
        st.warning('This analysis needs four conditions: control, drug A, drug B and the combination.'); st.stop()
    with st.container(border=True):
        roles = roles_input(key, conds)
        c1, c2 = st.columns([1, 2])
        with c1: t_end = t_end_input(key, t_max)
        with c2: window = window_input(key, t_end or t_max)
        names = {'v': 'velocity', 'lam': 'rate', 'L': 'length'}
        variables = st.multiselect('Variables', list(names), default=get(key, 'variables', ['v', 'lam', 'L']),
                                   format_func=names.get)
        other = st.text_input('Other windows to compare (optional)', value=get(key, 'other_windows_text', ''),
                              placeholder='e.g. 80-125, 95-170, 140-170',
                              help='The (α, β) table is repeated for these windows, to see whether the result depends on '
                                   'the window. Leave empty to skip.')
        with st.expander('More settings'):
            c1, c2, c3, c4 = st.columns(4)
            n_boot = c1.select_slider('Bootstrap resamples', [100, 200, 500, 1000], value=get(key, 'n_boot', 500))
            n_lev = c2.number_input('Quantile levels', 20, 500, int(get(key, 'n_levels', 100)), 10)
            v_min = c3.number_input('Lowest value used (control-median units)', 0.0, 1.0, float(get(key, 'v_min', 0.05)), 0.01)
            min_eff = c4.number_input('Smallest effect whose exponent is estimated', 0.0, 1.0,
                                      float(get(key, 'min_effect', 0.1)), 0.01)
            c1, c2, c3 = st.columns(3)
            um = c1.number_input('µm per pixel', 0.0, 10.0, float(get(key, 'um_per_px', 0) or 0), 0.001, format='%.4f',
                                 help='0 = keep pixels')
            cover = c2.slider('A mother counts if its intervals cover at least', 0.1, 1.0, float(get(key, 'cover', 0.6)), 0.05)
            dtr = c3.slider('Intervals between images allowed (min)', 0.0, 60.0, tuple(float(x) for x in get(key, 'dt_range', [3, 25])), 1.0)
        try:
            ow = [tuple(float(x) for x in w.split('-')) for w in re.split(r'[,;]', other) if w.strip()]
            assert all(len(w) == 2 and w[0] < w[1] for w in ow)
        except Exception:
            st.error("Write the other windows as 'start-end' separated by commas, e.g. 80-125, 95-170"); st.stop()
        if not variables:
            st.warning('Choose at least one variable.'); st.stop()
        sets = dict(roles=roles, window=list(window), t_end=t_end, variables=variables, other_windows_text=other,
                    n_boot=n_boot, n_levels=n_lev, v_min=v_min, min_effect=min_eff, um_per_px=um or None, cover=cover,
                    dt_range=list(dtr))
        run_box(key, 'quantiles', mm.run_quantile_bliss,
                dict(experiments=entries, roles=roles, window=window, t_end=t_end, variables=variables, other_windows=ow,
                     cover=cover, dt_range=tuple(dtr), um_per_px=um or None, n_levels=n_lev, v_min=v_min,
                     min_effect=min_eff, n_boot=n_boot), sets)

    def tabs(r):
        out = [('(α, β): how much of each drug is in the combination', r.tables['alpha_beta'].T.astype(str)),
               ('The effects as vectors: angles (°) and distances', r.tables['vectors_3d'].round(3).T)]
        if 'alpha_beta_windows' in r.tables:
            out.append(('(α, β) in other windows', r.tables['alpha_beta_windows'].astype(str)))
        return out
    show_run(key, tabs)


def page_help():
    st.title('Help')
    st.markdown("""
**What this is.** A point-and-click version of the analysis notebooks of this folder. It reads the Excel sheets of
the mother cells written by the mother machine pipeline (`Stacks/Excel/<area>-<date>_longitud_madres.xlsx`) and
runs the same analyses with the same code.

**How to use it**

1. **1 · Experiments**: choose each folder of Excel files and say which condition each group of areas is (and, if
   you know it, when the flow started: that is time 0). The page checks that everything can be read.
2. Choose an analysis in the panel on the left, adjust the settings (each one has a ⓘ with an explanation) and
   press **▶ Run**.
3. The figures and tables appear below and are saved in the **results folder** (panel on the left), in
   `<analysis>/<run name>/`: figures as PNG (200 dpi, sized for slides), tables as CSV (open them with Excel), and
   `settings.json` with everything needed to repeat the run (**↩ Load a previous run**).

The experiments and the last settings of each analysis are remembered the next time you open the interface.

**The analyses**

| page | what it answers |
|---|---|
| 2 · Division phenotypes | which division patterns there are, and how often each condition shows each one |
| 3 · Length phenotypes | the same with the length (filaments, arrested cells) |
| 4 · Rate × length and Bliss | does a drug change the velocity through the rate or through the length; is the combination what Bliss predicts |
| 5 · Bliss on quantiles | the same on the whole distributions; how much of each drug is in the combination (α, β) |

The notebooks (`01_…ipynb` to `04_…ipynb`) explain every step and every formula in detail.

**Conventions.** Time 0 is the flow start if given, otherwise the first image of the folder. A mother is lost when
its top cell changes without a division (a comment in the Excel) or the channel empties; lost mothers are left out
where the whole trajectory is needed. Intervals (95 %) are bootstraps over **areas**: with one experiment per
condition they do not include the variation between days.

**Problems.** If the interface does not react, look at the black window that opened with it (errors appear there).
To stop the interface, close that window.
""")


# ============================================================================================== dispatch
try:
    {PAGES[0]: page_experiments, ANALYSES['divisions']['page']: page_divisions, ANALYSES['lengths']['page']: page_lengths,
     ANALYSES['growth']['page']: page_growth, ANALYSES['quantiles']['page']: page_quantiles}.get(page, page_help)()
finally:
    st.session_state.last_page = page
