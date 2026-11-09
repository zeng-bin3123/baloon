#!/bin/bash
# Unit p17652417b0 (python tooling, pytools/). Runs the invariant test (tests/k37_pytools/test_pytools.py: port table,
# code-plane packing, canonical quant forms vs the reference dequantisers, GGUF IO, expert blob layout, MTP quantiser vs
# scalar models, shard fetching over local files and HTTP, IQ index, byte-level BPE), then mutation self-tests proving
# that deliberately broken copies of pytools/ are caught.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${BALOON_TEST_DIR:-/tmp/baloon_unit_p176_test}"
rm -rf "$OUT"; mkdir -p "$OUT"
export PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=ignore
command -v python3 >/dev/null 2>&1 || { echo "python3 missing"; exit 2; }
python3 -c "import numpy" || { echo "numpy missing"; exit 2; }
rc=0

echo "== unit test =="
PYTOOLS_ROOT="$ROOT" python3 tests/k37_pytools/test_pytools.py >"$OUT/base.log" 2>&1 || rc=1
tail -4 "$OUT/base.log"
grep -q "^OK" "$OUT/base.log" || rc=1
[ "$(grep -c '^Ran [0-9]* tests' "$OUT/base.log")" = 1 ] || rc=1

echo "== mutation self-tests (each mutant must be caught) =="
mutant() {  # mutant <name> <file under pytools/> <old> <new>
  local name="$1" f="$2" old="$3" new="$4" m="$OUT/m_$1"
  rm -rf "$m"; mkdir -p "$m/.einsteinOS"; cp -r pytools "$m/"; cp -r .einsteinOS/port "$m/.einsteinOS/"
  python3 - "$m/pytools/$f" "$old" "$new" <<'PY' || { echo "mutant $name: pattern not found"; rc=1; return; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if s.count(old) != 1: sys.exit(1)
open(p, "w").write(s.replace(old, new))
PY
  if PYTOOLS_ROOT="$m" timeout 600 python3 tests/k37_pytools/test_pytools.py >"$m/log" 2>&1; then
    echo "mutant $name SURVIVED"; rc=1
  else
    echo "mutant $name killed"
  fi
}
mutant packorder   planes.py      'out |= c[:, k] << (bits * k)' 'out |= c[:, k] << (bits * (per - 1 - k))'
mutant pad         planes.py      'return (ALIGN - n % ALIGN) % ALIGN' 'return ALIGN - n % ALIGN'
mutant q6hi        canon_xcheck.py 'codes[:, e + 32:e + 64] = (L[:, 32:] & 0xF) | (((H >> 2) & 3) << 4)' 'codes[:, e + 32:e + 64] = (L[:, 32:] & 0xF) | (((H >> 2) & 3) << 3)'
mutant bias        canon_xcheck.py '(c.codes.astype(np.int32) - s.bias)' '(c.codes.astype(np.int32) - s.bias + 1)'
mutant q3mask      canon_xcheck.py 'm = 1 << (4 * h + j)' 'm = 1 << j'
mutant scalemin    canon_xcheck.py '(q[..., j + 4] & 0xF) | ((q[..., j - 4] >> 6) << 4)' '(q[..., j + 4] & 0xF) | ((q[..., j - 4] >> 5) << 4)'
mutant q50high     canon_xcheck.py '((qh[:, None] >> (j + 12)) & 0x10)' '((qh[:, None] >> (j + 11)) & 0x10)'
mutant nibsplit    canon_xcheck.py 'return np.concatenate([qs & 0xF, qs >> 4], axis=1)' 'return np.concatenate([qs >> 4, qs & 0xF], axis=1)'
mutant bf16        canon_xcheck.py 'astype(np.uint32) << np.uint32(16)' 'astype(np.uint32) << np.uint32(15)'
mutant interleave  layer_pack.py  'np.stack([gs, us], axis=1)' 'np.stack([us, gs], axis=1)'
mutant planeoff    layer_pack.py  '("d_codes", D_CODES), ("d_scales", D_SCALES)' '("d_scales", D_SCALES), ("d_codes", D_CODES)'
mutant q4sign      mtp_quant.py   'np.float32(-8.0)' 'np.float32(8.0)'
mutant q2cands     mtp_quant.py   'np.float32((16 + k) / 32.0)' 'np.float32((15 + k) / 32.0)'
mutant q8scale     mtp_quant.py   'np.float32(127.0)' 'np.float32(128.0)'
mutant q2tie       mtp_quant.py   'better = (best_err < 0) | (err < best_err)' 'better = (best_err < 0) | (err <= best_err)'
mutant actround    mtp_blob.py    'np.clip(np.rint(c * idv[:, None]), -127, 127)' 'np.clip(np.trunc(c * idv[:, None]), -127, 127)'
mutant bytemap     bpe_extract.py 'm[b] = chr(256 + n)' 'm[b] = chr(257 + n)'
mutant bpeorder    bpe_extract.py 'if r is not None and (best is None or r < best):' 'if r is not None and (best is None or r > best):'
mutant dataoff     gguf_file.py   'self.data_offset = (c.pos + self.alignment - 1) // self.alignment * self.alignment' 'self.data_offset = c.pos'
mutant writerpad   gguf_emit.py   'o += b"\0" * _pad(a, len(o))' 'o += b""'
mutant q2deq       gguf_emit.py   '(d[:, None] * (codes - 1).astype(np.float32))' '(d[:, None] * (codes - 2).astype(np.float32))'
mutant expertname  iq_index.py    '".experts." in name or "_exps" in name' '".experts." in name or "_expz" in name'
mutant refbits     ref_pack.py    'return cx.MAPPINGS[n].bits' 'return cx.MAPPINGS[n].bits // 2'
mutant refscale    ref_pack.py    'return "f32"' 'return "f16"'
mutant refdigest   ref_pack.py    'if sha256(_slice(e, blob, role)) != p["sha256"]:' 'if False:'
mutant refalign    ref_pack.py    'if p["offset"] % planes.ALIGN:' 'if False:'
mutant refexpflag  ref_pack.py    'if e["expert"] != is_expert(e["name"]):' 'if False:'
mutant refdup      ref_pack.py    '        if name in seen:' '        if False:'
mutant modelbytes  iq_index.py    'out[e["file"]] += e["nbytes"]' 'out[e["file"]] += e["nbytes"] + 1'
mutant overlay     iq_index.py    'merged.update(o)' 'merged.update({k: v for k, v in o.items() if k not in merged})'
mutant rangeslice  mtp_download.py 'data = data[start:end]' 'data = data'
mutant hdrstart    mtp_download.py 'return hdr, 8 + n' 'return hdr, 8'
mutant arenawhole  pack_catalog.py 'if arena % layer_bytes:' 'if False:'
mutant gguf_py     locate.py      'if (c / "gguf" / "__init__.py").is_file():' 'if c.is_dir():'

[ "$rc" = 0 ] && echo "ALL TESTS PASSED" || echo "TESTS FAILED"
exit $rc
