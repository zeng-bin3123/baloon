#!/usr/bin/env python3
# check_port.py - every row of .einsteinOS/port/u63af4bc057.tsv names a file that exists and DEFINES the listed name
# (macro, type, constant, function, template specialisation or kernel). Exits 1 on any missing row.
import re, sys, os
root = sys.argv[1]
rows = [l.rstrip('\n').split('\t') for l in open(os.path.join(root, '.einsteinOS/port/u63af4bc057.tsv')) if l.strip()]
bad = 0
for r in rows:
    if len(r) != 4: print('MALFORMED', r); bad += 1; continue
    _, _, tf, tn = r
    p = os.path.join(root, tf)
    if not os.path.exists(p): print('NOFILE', tf); bad += 1; continue
    txt = open(p, errors='ignore').read()
    if '::' in tn:
        pats = [re.escape(tn) + r'\s*\([^;{]*\)\s*(const\s*)?\{']
    else:
        n = re.escape(tn)
        pats = [r'#\s*define\s+' + n + r'\b', r'\b(struct|class|enum(\s+class)?|union)\s+' + n + r'\b', r'\busing\s+' + n + r'\s*=',
                r'\b' + n + r'\s*(\[[^\]]*\])*\s*(=|\{)', r'\b' + n + r'\s*(<[^<>()]*>)?\s*\([^;{]*\)\s*(const\s*)?(noexcept\s*)?(->[^{;]*)?\{']
    if not any(re.search(q, txt, re.S) for q in pats): print('NODEF', tf, tn); bad += 1
print('port rows: %d, missing: %d' % (len(rows), bad))
sys.exit(1 if bad or len(rows) != 212 else 0)
