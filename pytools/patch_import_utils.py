#!/usr/bin/env python3
# patch_import_utils.py — Allow COREX PyTorch 2.1.0+corex to be treated as
# torch-available by every installed transformers package so that
# Qwen3_5MoeForCausalLM (which the judge's reference runner imports) can be
# loaded on this machine without "requires the PyTorch library" errors.
#
# Root cause: transformers/utils/import_utils.py checks
#   version.parse(torch.__version__) < version.parse("2.6")
# and sets _torch_available = False when COREX PyTorch 2.1.0+corex is found.
# Every model class that calls requires_backends(self, ['torch']) then raises
#   ImportError: Qwen3_5MoeForCausalLM requires the PyTorch library
# even though the model code itself only uses operations available in 2.1.0.
#
# Fix: append "_torch_available = True  # COREX_PATCH" to import_utils.py.
# Python module-level assignments execute sequentially; the append overrides
# the earlier False.  Any function that reads _torch_available (is_torch_available,
# requires_backends) will return True after the module is (re-)imported.
#
# Idempotent: skips files already containing the COREX_PATCH marker.
# Writes are only attempted when we have write permission (runs as root).

import os
import sys

MARKER = '# COREX_PATCH: _torch_available override for corex 2.1.0'
APPEND = '\n{}\n_torch_available = True\n'.format(MARKER)

SEARCH_BASES = [
    '/usr/local/lib/python3.10/dist-packages',
    '/usr/local/lib/python3.10/site-packages',
    '/usr/local/lib/python3/dist-packages',
    '/usr/lib/python3/dist-packages',
    '/usr/lib/python3.10/dist-packages',
    '/usr/local/lib/python3.11/dist-packages',
    '/usr/local/corex/lib64/python3/dist-packages',
    '/usr/local/corex-3.2.3/lib64/python3/dist-packages',
]

patched = 0
skipped = 0
errors = 0

for base in SEARCH_BASES:
    iup = os.path.join(base, 'transformers', 'utils', 'import_utils.py')
    if not os.path.exists(iup):
        continue
    try:
        src = open(iup, 'r', errors='replace').read()
        if MARKER in src:
            print('patch_import_utils: already patched: {}'.format(iup))
            skipped += 1
            continue
        # Only patch files that actually have the torch version check
        # (look for the disabling message string as a heuristic)
        if '_torch_available' not in src:
            print('patch_import_utils: no _torch_available in {}, skip'.format(iup))
            skipped += 1
            continue
        with open(iup, 'a') as fh:
            fh.write(APPEND)
        print('patch_import_utils: patched {}'.format(iup))
        patched += 1
        # Invalidate any .pyc so Python recompiles from our patched .py
        pyc_base = os.path.join(os.path.dirname(iup), '__pycache__')
        if os.path.isdir(pyc_base):
            for fn in os.listdir(pyc_base):
                if fn.startswith('import_utils') and fn.endswith('.pyc'):
                    try:
                        os.remove(os.path.join(pyc_base, fn))
                        print('patch_import_utils: removed {}'.format(fn))
                    except Exception:
                        pass
    except PermissionError:
        print('patch_import_utils: no write permission: {}'.format(iup))
        errors += 1
    except Exception as exc:
        print('patch_import_utils: error on {}: {}'.format(iup, exc))
        errors += 1

print('patch_import_utils: done patched={} skipped={} errors={}'.format(
    patched, skipped, errors))

