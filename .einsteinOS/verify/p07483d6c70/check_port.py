# check_port.py  —  unit p07483d6c70
# Usage: python3 check_port.py <tsv> <kinds_txt>
# Reads every row of <tsv> (ref_file TAB ref_name TAB dev_file TAB dev_name)
# and asserts that dev_name is defined in dev_file using the kind from <kinds_txt>.
# kinds_txt format: ref_file TAB ref_name TAB kind   (kind = type | const | func)
# Exits 0 when all 40 rows pass, 1 otherwise.
import re, sys

kinds = {}
if len(sys.argv) > 2:
    for line in open(sys.argv[2]):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3:
            kinds[(p[0], p[1])] = p[2]

def strip(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)

def pat(name, kind):
    e = re.escape(name)
    if kind == "type":
        return r"\b(?:struct|class)\s+" + e + r"\b"
    if kind == "const":
        return r"\b" + e + r"\s*(?:=|\{)"
    return r"\b" + e + r"\s*\("

bad = 0
n = 0
seen = set()
for line in open(sys.argv[1]):
    row = line.rstrip("\r\n").split("\t")
    if len(row) != 4 or not row[0]:
        print("port: malformed row", repr(line))
        bad += 1
        continue
    n += 1
    key = (row[0], row[1])
    seen.add(key)
    try:
        text = strip(open(row[2], errors="ignore").read())
    except OSError:
        print("port: missing file", row[2])
        bad += 1
        continue
    kind = kinds.get(key, "func")
    if not re.search(pat(row[3], kind), text, re.M | re.S):
        print("port: %s not found in %s  (kind=%s)" % (row[3], row[2], kind))
        bad += 1

EXPECTED = 40
if n != EXPECTED or len(seen) != EXPECTED:
    print("port table has %d rows (%d distinct), expected %d" % (n, len(seen), EXPECTED))
    bad += 1

print("port rows: %d  bad: %d" % (n, bad))
sys.exit(1 if bad else 0)
