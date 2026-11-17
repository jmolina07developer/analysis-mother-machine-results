"""Check that everything the analyses need is installed and works on this computer (it does not modify anything).

Usage (with the environment activated):  python check_install.py
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TESTED = {'numpy': '2.2.6', 'pandas': '2.3.3', 'scipy': '1.15.3', 'sklearn': '1.7.2', 'openpyxl': '3.1.5',
          'matplotlib': '3.10.9', 'streamlit': '1.65.0'}
PIP = {'sklearn': 'scikit-learn'}
problems = []
ok = lambda m: print(f'  [ok]   {m}')
warn = lambda m: print(f'  [warn] {m}')


def fail(m):
    print(f'  [FAIL] {m}'); problems.append(m)


print('1. Python')
v = sys.version_info
if v < (3, 10):
    fail(f'Python {v.major}.{v.minor} is too old: use Python 3.10-3.12')
else:
    ok(f'Python {v.major}.{v.minor}.{v.micro}')
print('2. Packages')
for mod, ver in TESTED.items():
    try:
        m = importlib.import_module(mod)
        got = getattr(m, '__version__', '?')
        (ok if got == ver else warn)(f'{PIP.get(mod, mod)} {got}' + ('' if got == ver else f' (tested with {ver})'))
    except Exception as e:
        fail(f'{PIP.get(mod, mod)} is missing or broken: {e}')
print('3. The analyses')
try:
    import mmtools as mm
    ok(f"mmtools: {', '.join(mm.ANALYSES)}")
except Exception as e:
    fail(f'mmtools cannot be imported: {e}')
print('4. Folder dialogs (tkinter)')
try:
    import tkinter  # noqa: F401
    ok('tkinter')
except Exception:
    warn('tkinter is missing: the "Choose…" buttons will not work, paste the paths instead')
print()
if problems:
    print(' Something is missing: run the installer again (install_mac.command / install_windows.bat).')
    sys.exit(1)
print(' Everything is ready.')
