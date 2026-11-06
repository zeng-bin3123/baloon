"""Unit p17652417b0: invariants of the python tooling (code-plane packing, canonical quant forms, GGUF IO, expert blob
layout, MTP quantiser, shard fetching, IQ index, byte-level BPE). Exits non-zero on any failed check.
Run with PYTOOLS_ROOT (default: repo root) naming the tree whose `pytools` package is under test."""
import json
import os
import re
import struct
import sys
import tempfile
import threading
import unittest
import functools
import http.server

ROOT = os.path.abspath(os.environ.get("PYTOOLS_ROOT") or os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402

from pytools import (bpe_extract, canon_xcheck as cx, gguf_emit, gguf_file as gf, iq_index, layer_pack as lp,  # noqa: E402
                     locate, mtp_blob, mtp_download as dl, mtp_quant as mq, pack_catalog, planes)

TSV = os.path.join(ROOT, ".einsteinOS", "port", "p17652417b0.tsv")
RNG = np.random.default_rng(20261105)


def tmp(suffix=""):
    fd, p = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return p


class PortTable(unittest.TestCase):
    def test_rows_defined(self):
        rows = [l.rstrip("\r\n").split("\t") for l in open(TSV) if l.strip()]
        self.assertEqual(len(rows), 72)
        self.assertEqual(len({(r[0], r[1]) for r in rows}), 72)
        for rf, rn, mf, mn in rows:
            src = open(os.path.join(ROOT, mf)).read()
            self.assertRegex(src, r"(?m)^(def|class) %s\b" % re.escape(mn), "%s not defined in %s" % (mn, mf))


class Planes(unittest.TestCase):
    def test_roundtrip_all_widths(self):
        for bits in (2, 4, 8):
            per = 8 // bits
            c = RNG.integers(0, 1 << bits, per * 100)
            p = planes.pack_codes(c, bits)
            self.assertEqual(len(p), 100)
            self.assertTrue(np.array_equal(planes.unpack_codes(p, bits), c))

    def test_lsb_first(self):
        self.assertEqual(planes.pack_codes([1, 2, 3, 0], 2).tolist(), [1 | (2 << 2) | (3 << 4)])
        self.assertEqual(planes.pack_codes([0xA, 0x5], 4).tolist(), [0x5A])
        self.assertEqual(planes.pack_codes([7], 8).tolist(), [7])

    def test_rejects(self):
        with self.assertRaises(ValueError): planes.pack_codes([4], 2)
        with self.assertRaises(ValueError): planes.pack_codes([1, 2, 3], 2)
        with self.assertRaises(ValueError): planes.pack_codes([1], 3)

    def test_pad64(self):
        for n in range(0, 300):
            self.assertEqual((n + planes.pad64(n)) % 64, 0)
            self.assertLess(planes.pad64(n), 64)
        self.assertEqual(planes.pad64(128), 0)
        w = planes.PlaneWriter()
        offs = [w.append_plane(b"x" * n) for n in (5, 64, 100, 1)]
        self.assertTrue(all(o % 64 == 0 for o in offs))
        self.assertEqual(offs, [0, 64, 128, 256])


def rand_blocks(name, n=6):
    s = cx.MAPPINGS[name]
    raw = bytearray(RNG.integers(0, 256, s.block_bytes * n, dtype=np.uint8).tobytes())
    scale_at = {"Q2_0": [0], "Q4_0": [0], "IQ4_NL": [0], "Q5_0": [0], "Q8_0": [0], "Q6_K": [208],
                "Q4_K": [0, 2], "Q5_K": [0, 2], "Q3_K": [108], "IQ4_XS": [0]}[name]
    for b in range(n):
        for o in scale_at:
            raw[b * s.block_bytes + o:b * s.block_bytes + o + 2] = np.float16(RNG.standard_normal() * 0.2).tobytes()
    return bytes(raw)


class Canonical(unittest.TestCase):
    def test_every_type_decodes_bit_exact(self):
        for name in cx.MAPPINGS:
            for _ in range(3):
                self.assertEqual(cx.check_block_set(name, rand_blocks(name)), [], name)

    def test_named_functions(self):
        pairs = [("Q2_0", cx.to_s2, cx.from_s2), ("Q4_0", cx.to_s4_q4_0, cx.from_s4_q4_0),
                 ("IQ4_NL", cx.to_s4_iq4_nl, cx.from_s4_iq4_nl), ("IQ4_XS", cx.to_s4_iq4_xs, cx.from_s4_iq4_xs),
                 ("Q3_K", cx.to_s4_q3_k, cx.from_s4_q3_k), ("Q4_K", cx.to_s4_q4_k, cx.from_s4_q4_k),
                 ("Q5_0", cx.to_s8_q5_0, cx.from_s8_q5_0), ("Q5_K", cx.to_s8_q5_k, cx.from_s8_q5_k),
                 ("Q6_K", cx.to_s8_q6_K, cx.from_s8_q6_K), ("Q8_0", cx.to_s8_q8_0, cx.from_s8_q8_0)]
        for name, to, frm in pairs:
            raw = rand_blocks(name)
            c = to(raw)
            s = cx.MAPPINGS[name]
            self.assertEqual(c.codes.shape, (6, s.block_elems))
            self.assertLess(int(c.codes.max()), 1 << s.bits)               # codes fit their plane width
            ref = cx.reference_values(name, raw)
            self.assertTrue(np.array_equal(frm(c).reshape(-1).view(np.uint32), ref.view(np.uint32)), name)

    def test_bias_is_integer_and_before_multiply(self):
        # all-zero codes decode to -bias * scale exactly
        for name, bias in (("Q4_0", 8), ("Q5_0", 16), ("Q6_K", 32), ("Q2_0", 1), ("Q3_K", 4), ("Q8_0", 128)):
            s = cx.MAPPINGS[name]
            self.assertEqual(s.bias, bias)
            c = cx.Canon(np.zeros((1, s.block_elems), np.uint8), np.full((1, s.block_elems // s.group_elems), 0.5, np.float32),
                         np.zeros((1, s.block_elems // s.group_elems), np.float32))
            self.assertTrue((cx._decode(name, c) == np.float32(-bias * 0.5)).all())
        # code ranges decode to -8..7 etc.
        for name, maxcode, lo, hi in (("Q4_0", 15, -8, 7), ("Q5_0", 31, -16, 15), ("Q8_0", 255, -128, 127),
                                      ("Q2_0", 3, -1, 2), ("Q3_K", 7, -4, 3), ("Q6_K", 63, -32, 31)):
            self.assertEqual((0 - cx.MAPPINGS[name].bias, maxcode - cx.MAPPINGS[name].bias), (lo, hi))

    def test_geometry_table(self):
        for name, s in cx.MAPPINGS.items():
            self.assertEqual(gf.block_geometry(s.type), (s.block_elems, s.block_bytes), name)
            self.assertEqual(s.block_elems % s.group_elems, 0)
        self.assertEqual(2 + 16, 18)
        self.assertEqual({n: cx.MAPPINGS[n].block_bytes for n in ("Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ4_XS")},
                         {"Q3_K": 110, "Q4_K": 144, "Q5_K": 176, "Q6_K": 210, "IQ4_XS": 136})

    def test_iq4_codebook(self):
        kv = cx.KVALUES_IQ4NL.astype(int).tolist()
        self.assertEqual(len(kv), 16)
        self.assertTrue(all(a < b for a, b in zip(kv, kv[1:])))
        self.assertNotEqual(kv[1] - kv[0], kv[2] - kv[1])
        self.assertEqual(kv[0], -127); self.assertEqual(kv[15], 113)

    def test_scale_min_k4(self):
        q = RNG.integers(0, 256, (50, 12), dtype=np.uint8)
        for j in range(8):
            d, m = cx.get_scale_min_k4(j, q)
            for i in range(50):
                row = [int(v) for v in q[i]]
                if j < 4:
                    ed, em = row[j] & 63, row[j + 4] & 63
                else:
                    ed, em = (row[j + 4] & 0xF) | ((row[j - 4] >> 6) << 4), (row[j + 4] >> 4) | ((row[j] >> 6) << 4)
                self.assertEqual((int(d[i]), int(m[i])), (ed, em))
                self.assertLess(int(d[i]), 64); self.assertLess(int(m[i]), 64)

    def test_q3k_scales(self):
        p = RNG.integers(0, 256, (40, 12), dtype=np.uint8)
        got = cx.unpack_q3_k_scales(p)
        self.assertEqual(got.shape, (40, 16)); self.assertEqual(got.dtype, np.int8)
        for i in range(40):                      # scalar model: low nibbles/high 2-bit fields of the 12-byte layout
            r = [int(v) for v in p[i]]
            for k in range(16):
                low = (r[k] & 0xF) if k < 8 else (r[k - 8] >> 4)
                hi = (r[8 + (k % 4)] >> (2 * (k // 4))) & 3
                self.assertEqual(int(got[i, k]) & 0xFF, low | (hi << 4), (i, k))

    def test_q3k_mask_bits_distinct(self):
        masks = [1 << (4 * h + j) for h in range(2) for j in range(4)]
        self.assertEqual(sorted(masks), [1, 2, 4, 8, 16, 32, 64, 128])

    def test_s2_lossless(self):
        raw = rand_blocks("Q2_0", 20)
        self.assertEqual(cx.s2_to_raw(cx.to_s2(raw)), raw)

    def test_passthrough(self):
        h = RNG.standard_normal(64).astype(np.float16)
        raw = h.tobytes()
        self.assertTrue(np.array_equal(cx.from_p16_f16(cx.to_p16_f16(raw)), h.astype(np.float32)))
        f = RNG.standard_normal(64).astype(np.float32)
        self.assertTrue(np.array_equal(cx.from_p32_f32(cx.to_p32_f32(f.tobytes())), f))
        bf = (f.view(np.uint32) >> 16).astype("<u2")
        w = cx.to_p32_bf16(bf.tobytes())
        self.assertEqual(w.dtype, np.uint32)
        self.assertTrue(((w & 0xFFFF) == 0).all())
        self.assertTrue(np.array_equal(cx.from_p32_bf16(w).view(np.uint32), bf.astype(np.uint32) << 16))
        self.assertEqual(cx.p32_bf16_to_raw(w), bf.tobytes())

    def test_checker_can_fail(self):
        raw = rand_blocks("Q6_K")
        c = cx.to_s8_q6_K(raw)
        bad = cx.Canon(c.codes.copy(), c.scales, c.offsets)
        bad.codes[0, 5] ^= 1
        ref = cx.reference_values("Q6_K", raw)
        self.assertFalse(np.array_equal(cx.from_s8_q6_K(bad).reshape(-1), ref))

    def test_names_and_offsets(self):
        self.assertEqual(cx.reference_name(2), "Q4_0"); self.assertEqual(cx.reference_name("q2_0"), "Q2_0")
        self.assertEqual(cx.reference_name(14), "Q6_K")
        p = tmp(".gguf")
        w = gguf_emit.GGUFWriter(alignment=64)
        w.add_tensor("t", (64,), 42, rand_blocks_q2())
        w.write(p)
        self.assertEqual(cx.data_section_offset(p) % 64, 0)
        with cx.open_shard(p) as g:
            self.assertEqual(cx.data_section_offset(g), g.data_offset)
        os.unlink(p)

    def test_run_full_on_real_file(self):
        w = gguf_emit.GGUFWriter()
        for name in cx.MAPPINGS:
            s = cx.MAPPINGS[name]
            w.add_tensor("w." + name, (s.block_elems * 2,), s.type, rand_blocks(name, 2))
        w.add_tensor("f16", (64,), 1, RNG.standard_normal(64).astype(np.float16).tobytes())
        w.add_tensor("bf16", (64,), 30, (RNG.standard_normal(64).astype(np.float32).view(np.uint32) >> 16).astype("<u2").tobytes())
        w.add_tensor("f32", (8,), 0, RNG.standard_normal(8).astype(np.float32).tobytes())
        p = tmp(".gguf"); w.write(p)
        n, fails = cx.run_full(p, blocks=2)
        os.unlink(p)
        self.assertEqual(n, len(cx.MAPPINGS) + 3); self.assertEqual(fails, [])


def rand_blocks_q2():
    return rand_blocks("Q2_0", 1)


class GGUFIO(unittest.TestCase):
    def test_roundtrip_and_alignment(self):
        for align in (32, 64, 128):
            w = gguf_emit.GGUFWriter(alignment=align)
            w.add_kv("general.architecture", "x"); w.add_kv("n", 5); w.add_kv("big", 1 << 40); w.add_kv("f", 1.5); w.add_kv("b", False)
            w.add_kv("toks", ["a", "bb", "ccc"]); w.add_kv("ids", [1, -2, 3])
            a, b = rand_blocks_q2() * 2, np.arange(7, dtype=np.float32).tobytes()
            w.add_tensor("a", (128,), 42, a); w.add_tensor("b", (7,), 0, b)
            p = tmp(); w.write(p)
            with gf.GGUFFile(p) as g:
                self.assertEqual(g.alignment, align); self.assertEqual(g.data_offset % align, 0)
                self.assertEqual([t.offset % align for t in g.tensors], [0, 0])
                self.assertEqual(g.metadata["n"], 5); self.assertEqual(g.metadata["big"], 1 << 40)
                self.assertAlmostEqual(g.metadata["f"], 1.5); self.assertIs(g.metadata["b"], False)
                self.assertEqual(g.metadata["toks"], ["a", "bb", "ccc"]); self.assertEqual(list(g.metadata["ids"]), [1, -2, 3])
                self.assertEqual(g.tensor_bytes("a"), a); self.assertEqual(g.tensor_bytes("b"), b)
                self.assertEqual(g.by_name["a"].nbytes, 36); self.assertEqual(g.by_name["a"].n_elements, 128)
            os.unlink(p)

    def test_rejects_bad_files(self):
        w = gguf_emit.GGUFWriter(); w.add_tensor("a", (64,), 42, rand_blocks_q2())
        data = bytearray(w.to_bytes())
        for cut in (3, 20, len(data) - 20):
            p = tmp(); open(p, "wb").write(bytes(data[:cut]))
            with self.assertRaises((gf.GGUFError, ValueError)): gf.GGUFFile(p).close()
            os.unlink(p)
        bad = bytearray(data); bad[:4] = b"GGUX"
        p = tmp(); open(p, "wb").write(bytes(bad))
        with self.assertRaises(gf.GGUFError): gf.GGUFFile(p).close()
        os.unlink(p)

    def test_writer_validation(self):
        w = gguf_emit.GGUFWriter()
        with self.assertRaises(ValueError): w.add_tensor("a", (64,), 42, b"\0" * 17)          # wrong byte count
        with self.assertRaises(ValueError): w.add_tensor("a", (63,), 42, b"\0" * 18)          # not whole blocks
        w.add_tensor("a", (64,), 42, b"\0" * 18)
        with self.assertRaises(ValueError): w.add_tensor("a", (64,), 42, b"\0" * 18)          # duplicate
        with self.assertRaises(ValueError): gguf_emit.GGUFWriter(alignment=0)

    def test_q2_quantiser(self):
        self.assertTrue(gguf_emit._selfcheck())
        x = RNG.standard_normal(64 * 20).astype(np.float32)
        raw = gguf_emit.quantize_q2_0(x)
        self.assertEqual(len(raw), 18 * 20)
        r = np.frombuffer(raw, np.uint8).reshape(-1, 18)
        codes = ((r[:, 2:, None] >> np.array([0, 2, 4, 6], np.uint8)) & 3)
        self.assertLessEqual(int(codes.max()), 3)
        d = np.ascontiguousarray(r[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
        self.assertTrue(np.array_equal(d, np.abs(x.reshape(-1, 64)).max(axis=1).astype(np.float16).astype(np.float32)))
        y = gguf_emit.dequantize_q2_0(raw).reshape(-1, 64)
        self.assertTrue(np.array_equal(y, d[:, None] * (codes.reshape(-1, 64).astype(np.int32) - 1).astype(np.float32)))
        # zero block -> scale 0 and symbol 0 (code 1) everywhere
        z = np.frombuffer(gguf_emit.quantize_q2_0(np.zeros(64, np.float32)), np.uint8)
        self.assertEqual(z.tolist(), [0, 0] + [0x55] * 16)


class ExpertBlob(unittest.TestCase):
    def test_geometry(self):
        self.assertEqual(lp.BLOB_BYTES, 1382400)
        self.assertEqual(3 * lp.ROLE_BYTES, lp.BLOB_BYTES)
        o = lp.blob_offsets()
        self.assertEqual([v[0] for v in o.values()], [0, 819200, 921600, 1331200])
        self.assertTrue(all(v[0] % 64 == 0 for v in o.values()))
        self.assertEqual(sum(v[1] for v in o.values()), lp.BLOB_BYTES)
        self.assertEqual(lp.FF * lp.SC_GU, lp.H * lp.SC_D)

    def test_build_roundtrip_and_interleave(self):
        g, u, d = (RNG.integers(0, 256, lp.ROLE_BYTES, dtype=np.uint8).tobytes() for _ in range(3))
        blob = lp.build(g, u, d)
        self.assertEqual(len(blob), lp.BLOB_BYTES)
        self.assertTrue(lp.verify(blob, g, u, d))
        self.assertEqual(lp.rebuild_role(blob, "gate"), g); self.assertEqual(lp.rebuild_role(blob, "up"), u)
        self.assertEqual(lp.rebuild_role(blob, "down"), d)
        # gate row r, block b, sits at slot 2r; up at 2r+1 (scale plane, then code plane)
        gs = np.frombuffer(g, np.uint8).reshape(lp.FF, lp.SC_GU, 18)
        us = np.frombuffer(u, np.uint8).reshape(lp.FF, lp.SC_GU, 18)
        bl = np.frombuffer(blob, np.uint8)
        so, co = lp.blob_offsets()["gu_scales"][0], lp.blob_offsets()["gu_codes"][0]
        for r, b in ((0, 0), (0, 39), (7, 3), (639, 39)):
            for k, src in ((0, gs), (1, us)):
                slot = 2 * r + k
                idx = slot * lp.SC_GU + b
                self.assertEqual(bl[so + 2 * idx:so + 2 * idx + 2].tolist(), src[r, b, :2].tolist())
                self.assertEqual(bl[co + 16 * idx:co + 16 * idx + 16].tolist(), src[r, b, 2:].tolist())
        # blob_of wraps build with a verification
        self.assertEqual(mtp_blob.blob_of(g, u, d), blob)
        self.assertEqual(mtp_blob.main([]), 0)
        # a changed byte in any role changes the blob and breaks verification against the originals
        d2 = bytearray(d); d2[12345] ^= 0xFF
        self.assertFalse(lp.verify(blob, g, u, bytes(d2)))

    def test_rejects_wrong_sizes(self):
        with self.assertRaises(ValueError): lp.build(b"\0" * 5, b"\0" * lp.ROLE_BYTES, b"\0" * lp.ROLE_BYTES)
        with self.assertRaises(ValueError): lp.rebuild_role(b"\0" * 10, "gate")
        with self.assertRaises(ValueError): lp.rebuild_role(b"\0" * lp.BLOB_BYTES, "bogus")

    def test_arena_addressing(self):
        self.assertEqual(lp.expert_offset(0, 0), 0)
        self.assertEqual(lp.expert_offset(1, 0), 512 * 1382400)
        self.assertEqual(lp.expert_offset(2, 7), (2 * 512 + 7) * 1382400)
        with self.assertRaises(ValueError): lp.expert_offset(0, 512)

    def test_q8_activation(self):
        x = (RNG.standard_normal(32 * 50) * np.float32(2.0 ** RNG.integers(-8, 8, 32 * 50))).astype(np.float32)
        x[:32] = 0; x[32:64] = 3.0
        q, d, s = mtp_blob.q8_0(x)
        c = x.reshape(-1, 32)
        for i in range(50):
            amax = np.float32(np.abs(c[i]).max()); dd = amax / np.float32(127)
            self.assertEqual(d[i], dd)
            idv = np.float32(1) / dd if dd != 0 else np.float32(0)
            exp = np.clip(np.rint(c[i] * idv), -127, 127).astype(np.int8)
            self.assertEqual(q[32 * i:32 * i + 32].tolist(), exp.tolist())
            self.assertEqual(int(s[i]), int(exp.astype(int).sum()))
        self.assertNotIn(-128, q.tolist())
        self.assertEqual(q.size, x.size)


def f32(v): return np.float32(v)


def scalar_q2(x):                     # independent scalar model of the Q2_0 scale search
    amax = f32(0)
    for v in x: amax = max(amax, abs(v))
    best, bh, bc = -1.0, None, None
    for k in range(17):
        h = np.float16(amax * f32((16 + k) / 32.0)); d = float(np.float32(h))
        codes, err = [], 0.0
        for v in x:
            s = 0
            if d != 0.0: s = max(-1, min(2, int(np.rint(float(v) / d))))
            codes.append(s + 1); e = float(v) - d * s; err += e * e
        if best < 0 or err < best: best, bh, bc = err, h, codes
    out = bytearray(np.float16(bh).tobytes()) + bytearray(16)
    for j, c in enumerate(bc): out[2 + j // 4] |= c << (2 * (j % 4))
    return bytes(out)


def scalar_q4(x):
    amax, mx = f32(0), f32(0)
    for v in x:
        if abs(v) > amax: amax, mx = abs(v), v
    d = mx / f32(-8.0); idv = f32(1) / d if d != 0 else f32(0)
    nib = lambda t: max(0, min(15, int(t)))
    out = bytearray(np.float16(d).tobytes())
    for j in range(16): out.append(nib(x[j] * idv + f32(8.5)) | (nib(x[j + 16] * idv + f32(8.5)) << 4))
    return bytes(out)


def scalar_q8(x):
    amax = f32(0)
    for v in x: amax = max(amax, abs(v))
    d = amax / f32(127.0); idv = f32(1) / d if d != 0 else f32(0)
    out = bytearray(np.float16(d).tobytes())
    for v in x: out.append(int(max(-127, min(127, int(np.rint(v * idv))))) & 0xFF)
    return bytes(out)


class MtpQuant(unittest.TestCase):
    def blocks(self, n, width):
        x = (RNG.standard_normal((n, width)) * 2.0 ** RNG.integers(-6, 6, (n, 1))).astype(np.float32)
        x[0] = 0; x[1, :] = 1.0; x[2, ::2] = -1.0; x[2, 1::2] = 1.0
        return x

    def test_against_scalar_models(self):
        x2, x4, x8 = self.blocks(40, 64), self.blocks(40, 32), self.blocks(40, 32)
        self.assertEqual(mq.q2_0(x2), b"".join(scalar_q2(r) for r in x2))
        self.assertEqual(mq.q4_0(x4), b"".join(scalar_q4(r) for r in x4))
        self.assertEqual(mq.q8_0(x8), b"".join(scalar_q8(r) for r in x8))

    def test_code_ranges_and_dequant(self):
        x = self.blocks(30, 64)
        raw = mq.q2_0(x); r = np.frombuffer(raw, np.uint8).reshape(-1, 18)
        d = np.ascontiguousarray(r[:, :2]).view(np.float16).astype(np.float32).reshape(-1)
        y = mq.dequant("q2_0", raw).reshape(-1, 64)
        for i in range(30):
            if d[i] != 0:
                s = np.rint(y[i] / d[i]); self.assertTrue(((s >= -1) & (s <= 2)).all())
        # search never does worse than the plain-amax scale (candidate k = 16 is in the set)
        base = np.frombuffer(gguf_emit.quantize_q2_0(x.reshape(-1)), np.uint8)
        eb = ((gguf_emit.dequantize_q2_0(base.tobytes()).reshape(-1, 64) - x) ** 2).sum(1)
        es = ((y - x) ** 2).sum(1)
        self.assertTrue((es <= eb * (1 + 1e-5) + 1e-12).all())
        for f in ("q4_0", "q8_0"):
            xx = self.blocks(30, 32); yy = mq.dequant(f, getattr(mq, f)(xx)).reshape(-1, 32)
            tol = {"q4_0": 1.01, "q8_0": 0.01}[f]
            self.assertTrue((np.abs(yy - xx).max(1) <= np.abs(xx).max(1) * tol / (2 if f == "q8_0" else 1) + 1e-6).all(), f)

    def test_totals_and_experts(self):
        self.assertEqual({f: mq.total_bytes(f) for f in mq.FORMATS},
                         {"q2_0": 707788800, "q4_0": 1415577600, "q8_0": 2673868800})
        self.assertEqual(512 * 1280 * (2560 // 64 * 18) + 512 * 2560 * (640 // 64 * 18), 707788800)
        w = RNG.standard_normal((3, 4, 128)).astype(np.float32)
        ex = mq._Experts(w)
        for f in mq.FORMATS:
            chunks = list(ex.quantize(f))
            self.assertEqual(len(chunks), 3); self.assertTrue(all(len(c) == ex.expert_bytes(f) for c in chunks))
            self.assertEqual(b"".join(chunks), getattr(mq, f)(w.reshape(-1, 128)))
            self.assertEqual(ex.total_bytes(f), 3 * ex.expert_bytes(f))
        with self.assertRaises(ValueError): mq._Experts(np.zeros((2, 3, 100), np.float32)).expert_bytes("q2_0")
        with self.assertRaises(ValueError): mq.q2_0(np.zeros(63, np.float32))

    def test_load_bf16(self):
        st = write_safetensors({"a": ("BF16", [2, 3], (np.arange(6, dtype=np.float32) * 1.5).view(np.uint32) >> 16)})
        v = mq.load_bf16(st, "a")
        self.assertEqual(v.shape, (2, 3))
        self.assertEqual(v.reshape(-1).tolist(), [0, 1.5, 3, 4.5, 6, 7.5])
        with self.assertRaises(KeyError): mq.load_bf16(st, "zz")
        os.unlink(st)


def write_safetensors(spec, path=None):
    hdr, blob = {}, b""
    for name, (dt, shape, arr) in spec.items():
        raw = np.asarray(arr).astype("<u2").tobytes() if dt in ("BF16", "F16") else np.asarray(arr).tobytes()
        hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    h = json.dumps(hdr).encode()
    path = path or tmp(".safetensors")
    with open(path, "wb") as f: f.write(struct.pack("<Q", len(h)) + h + blob)
    return path


class Handler(http.server.SimpleHTTPRequestHandler):       # ignores Range on purpose: get() must slice locally
    def log_message(self, *a): pass


class Fetch(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.t1 = (np.arange(8, dtype=np.float32) * 2).view(np.uint32) >> 16
        write_safetensors({"mtp.w": ("BF16", [8], self.t1), "other": ("BF16", [2], np.zeros(2, np.uint16))}, os.path.join(self.d, "s1.safetensors"))
        write_safetensors({"mtp.v": ("BF16", [2], np.ones(2, np.uint16))}, os.path.join(self.d, "s2.safetensors"))
        json.dump({"weight_map": {"mtp.w": "s1.safetensors", "other": "s1.safetensors", "mtp.v": "s2.safetensors"}},
                  open(os.path.join(self.d, "model.safetensors.index.json"), "w"))

    def test_local_get_header_inventory_fetch(self):
        f = os.path.join(self.d, "s1.safetensors")
        data = open(f, "rb").read()
        self.assertEqual(dl.get(f, 3, 11), data[3:11]); self.assertEqual(dl.get(f, 5), data[5:]); self.assertEqual(dl.get(f, 4, 4), b"")
        with self.assertRaises(IOError): dl.get(f, len(data) - 2, len(data) + 5)
        with self.assertRaises(ValueError): dl.get(f, 5, 2)
        hdr, start = dl.shard_header(f)
        self.assertEqual(start, 8 + struct.unpack("<Q", data[:8])[0])
        self.assertEqual(set(hdr), {"mtp.w", "other"})
        inv = dl.inventory(self.d)
        self.assertEqual(set(inv), {"mtp.w", "mtp.v"})
        w = inv["mtp.w"]
        self.assertEqual(w["end"] - w["begin"], 16); self.assertEqual(w["shape"], [8])
        self.assertEqual(dl.get(w["shard"], w["begin"], w["end"]), (np.asarray(self.t1).astype("<u2")).tobytes())
        self.assertEqual(set(dl.inventory(f, "other")), {"other"})
        dest = os.path.join(self.d, "copy.bin")
        self.assertEqual(dl.fetch(f, dest, chunk=7), len(data)); self.assertEqual(open(dest, "rb").read(), data)
        # resume from a partial file
        os.unlink(dest); open(dest + ".part", "wb").write(data[:20])
        self.assertEqual(dl.fetch(f, dest, chunk=5), len(data)); self.assertEqual(open(dest, "rb").read(), data)
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_http_ranges(self):
        h = functools.partial(Handler, directory=self.d)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            base = "http://127.0.0.1:%d" % srv.server_address[1]
            data = open(os.path.join(self.d, "s1.safetensors"), "rb").read()
            self.assertEqual(dl.get(base + "/s1.safetensors", 10, 30), data[10:30])
            self.assertEqual(dl.get(base + "/s1.safetensors", 10), data[10:])
            self.assertEqual(dl.shard_header(base + "/s1.safetensors")[1], dl.shard_header(os.path.join(self.d, "s1.safetensors"))[1])
            self.assertEqual(set(dl.inventory(base)), {"mtp.w", "mtp.v"})
            dest = os.path.join(self.d, "h.bin")
            self.assertEqual(dl.fetch(base + "/s1.safetensors", dest, chunk=13), len(data))
            self.assertEqual(open(dest, "rb").read(), data)
        finally:
            srv.shutdown()

    def test_main(self):
        self.assertEqual(dl.main(["inventory", self.d]), 0)
        self.assertEqual(dl.main(["bogus"]), 2)


class IqIndex(unittest.TestCase):
    def make(self, tensors):
        w = gguf_emit.GGUFWriter()
        for n, shape, t, data in tensors: w.add_tensor(n, shape, t, data)
        p = tmp(".gguf"); w.write(p); return p

    def test_expert_names_and_sizes(self):
        for n in ("blk.3.ffn_gate_exps.weight", "model.layers.0.mlp.experts.5.w", "mtp.experts.gate_up_proj"):
            self.assertTrue(iq_index.is_expert(n), n)
        for n in ("blk.3.ffn_gate_shexp.weight", "token_embd.weight", "blk.0.attn_q.weight"):
            self.assertFalse(iq_index.is_expert(n), n)
        self.assertEqual(iq_index.tensor_bytes(42, (64, 4)), 4 * 18)
        self.assertEqual(iq_index.tensor_bytes(23, (256, 2)), 2 * 136)
        self.assertEqual(iq_index.tensor_bytes(0, (3, 5)), 60)
        with self.assertRaises(ValueError): iq_index.tensor_bytes(23, (100,))

    def test_standalone_and_overlay(self):
        a = self.make([("blk.0.ffn_gate_exps.weight", (64, 2), 42, b"\1" * 36), ("token_embd.weight", (4,), 0, np.zeros(4, np.float32).tobytes())])
        b = self.make([("blk.0.ffn_gate_exps.weight", (64, 2), 2 and 42, b"\2" * 36)])
        s = iq_index.index_standalone(a)
        names = {e["name"]: e for e in s["tensors"]}
        self.assertEqual(set(names), {"blk.0.ffn_gate_exps.weight", "token_embd.weight"})
        self.assertTrue(names["blk.0.ffn_gate_exps.weight"]["expert"]); self.assertFalse(names["token_embd.weight"]["expert"])
        with gf.GGUFFile(a) as g:
            for e in s["tensors"]:
                raw = open(a, "rb").read()[e["offset"]:e["offset"] + e["nbytes"]]
                self.assertEqual(raw, g.tensor_bytes(e["name"]))
        m = iq_index.index_from_base(a, b)
        mn = {e["name"]: e for e in m["tensors"]}
        self.assertEqual(len(m["tensors"]), 2)
        self.assertEqual(mn["blk.0.ffn_gate_exps.weight"]["file"], 1); self.assertEqual(mn["token_embd.weight"]["file"], 0)
        out = tmp(".json")
        iq_index.write_index(out, m)
        self.assertEqual(iq_index.read_index(out), m)
        self.assertEqual(iq_index.main(["standalone", a, out]), 0)
        self.assertEqual(iq_index.main(["base", a, b, out]), 0)
        self.assertEqual(iq_index.main(["base", a]), 2)
        bad = json.loads(json.dumps(m)); bad["tensors"].append(dict(bad["tensors"][0]))
        with self.assertRaises(ValueError): iq_index.write_index(out, bad)
        bad = json.loads(json.dumps(m)); bad["tensors"][0]["nbytes"] += 1
        with self.assertRaises(ValueError): iq_index.validate(bad)
        c = self.make([("blk.0.ffn_gate_exps.weight", (128,), 42, b"\3" * 36)])
        with self.assertRaises(ValueError): iq_index.index_from_base(a, c)           # shape disagrees
        for p in (a, b, c, out): os.unlink(p)


class PackCatalog(unittest.TestCase):
    def test_describe(self):
        d = tempfile.mkdtemp()
        layer = 512 * lp.BLOB_BYTES
        with open(os.path.join(d, "experts.bin"), "wb") as f: f.truncate(2 * layer)
        open(os.path.join(d, "dense.bin"), "wb").write(b"\0" * 10)
        info = pack_catalog.describe(d)
        self.assertEqual(info["layers"], 2); self.assertEqual(info["blob_bytes"], 1382400)
        self.assertEqual(info["files"]["dense.bin"], 10)
        self.assertEqual(pack_catalog.main([d]), 0)
        self.assertEqual(json.load(open(os.path.join(d, "pack_index.json")))["layers"], 2)
        with open(os.path.join(d, "experts.bin"), "ab") as f: f.write(b"x")
        with self.assertRaises(ValueError): pack_catalog.describe(d)
        self.assertEqual(pack_catalog.main([d]), 1)
        self.assertEqual(pack_catalog.main([]), 2)


class Bpe(unittest.TestCase):
    def test_byte_map(self):
        m = bpe_extract.bytes_to_unicode()
        self.assertEqual(len(m), 256); self.assertEqual(len(set(m.values())), 256)
        printable = [b for b in range(256) if (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255)]
        self.assertEqual(len(printable), 188)
        for b in printable: self.assertEqual(ord(m[b]), b)
        nxt = 256
        for b in range(256):
            if b not in printable:
                self.assertEqual(ord(m[b]), nxt); nxt += 1
        self.assertEqual(nxt, 324)
        self.assertEqual(m[32], "\u0120"); self.assertEqual(m[10], "\u010a")

    def vocab(self):
        m = bpe_extract.bytes_to_unicode()
        toks = [m[b] for b in range(256)]
        g, h = m[ord("h")], m[ord("i")]
        merges = [g + " " + h, g + h + " " + m[ord("j")], h + " " + m[ord("j")]]    # (i, j) has a worse rank than (h, i)
        toks += [g + h, g + h + m[ord("j")], h + m[ord("j")], "<|eot|>", "<|user|>"]
        types = [1] * (len(toks) - 2) + [3, 4]
        return toks, merges, types

    def test_encode_decode(self):
        toks, merges, types = self.vocab()
        t = bpe_extract.Tokenizer(toks, merges, types)
        self.assertEqual(t.encode("hij"), [t.id_of["hij"]])
        self.assertEqual(len(t.encode("hi")), 1); self.assertEqual(len(t.encode("hx")), 2)
        for s in ("hij", "héllo wörld 123 \n\t!", "日本語 hi hij!", "", " "):
            self.assertEqual(t.decode(t.encode(s)), s)
        # one merge shortens the word by exactly one
        self.assertEqual(len(t._bpe("hi")), 1); self.assertEqual(len(t._bpe("hj")), 2)
        # special tokens: user-defined always matched, control only when parse_special
        self.assertEqual(t.encode("a<|user|>b").count(t.id_of["<|user|>"]), 1)
        self.assertNotIn(t.id_of["<|eot|>"], t.encode("<|eot|>"))
        self.assertEqual(t.encode("<|eot|>", parse_special=True), [t.id_of["<|eot|>"]])
        with self.assertRaises(ValueError): bpe_extract.Tokenizer(toks, merges, [1])

    def test_extract_from_gguf(self):
        toks, merges, types = self.vocab()
        w = gguf_emit.GGUFWriter()
        w.add_kv("tokenizer.ggml.model", "gpt2"); w.add_kv("tokenizer.ggml.tokens", toks)
        w.add_kv("tokenizer.ggml.merges", merges); w.add_kv("tokenizer.ggml.token_type", types)
        w.add_kv("tokenizer.ggml.eos_token_id", 258)
        p = tmp(".gguf"); w.write(p)
        d = bpe_extract.extract(p)
        self.assertEqual(d["tokens"], toks); self.assertEqual(d["merges"], merges); self.assertEqual(d["types"], types)
        self.assertEqual(d["eos_token_id"], 258)
        out = tmp(".json")
        self.assertEqual(bpe_extract.main([p, out]), 0)
        self.assertEqual(json.load(open(out))["tokens"], toks)
        self.assertEqual(bpe_extract.main([p, "--encode", "hij"]), 0)
        w2 = gguf_emit.GGUFWriter(); w2.add_kv("general.architecture", "x")
        p2 = tmp(".gguf"); w2.write(p2)
        with self.assertRaises(ValueError): bpe_extract.extract(p2)
        self.assertEqual(bpe_extract.main([p2]), 1)
        for q in (p, p2, out): os.unlink(q)


class Locate(unittest.TestCase):
    def test_gguf_py(self):
        d = tempfile.mkdtemp(); os.makedirs(os.path.join(d, "gg", "gguf"))
        open(os.path.join(d, "gg", "gguf", "__init__.py"), "w").write("")
        old = os.environ.get(locate.ENV_VAR)
        try:
            os.environ[locate.ENV_VAR] = os.path.join(d, "gg")
            self.assertEqual(str(locate.gguf_py()), os.path.join(d, "gg"))
            n = len(sys.path)
            self.assertEqual(str(locate.add_gguf_py()), os.path.join(d, "gg")); self.assertEqual(sys.path[0], os.path.join(d, "gg"))
            locate.add_gguf_py(); self.assertEqual(len(sys.path), n + 1)               # idempotent
            os.makedirs(os.path.join(d, "empty"))
            os.environ[locate.ENV_VAR] = os.path.join(d, "empty")          # exists, but holds no gguf package
            self.assertIsNone(locate.gguf_py(os.path.join(d, "empty")))
            self.assertIsNone(locate.add_gguf_py(os.path.join(d, "empty")))
            os.environ[locate.ENV_VAR] = os.path.join(d, "missing")
            self.assertIsNone(locate.gguf_py(d + "/gg/gguf"))
        finally:
            if old is None: os.environ.pop(locate.ENV_VAR, None)
            else: os.environ[locate.ENV_VAR] = old
            if d + "/gg" in sys.path: sys.path.remove(d + "/gg")
        os.makedirs(os.path.join(d, "up", "gguf-py", "gguf")); open(os.path.join(d, "up", "gguf-py", "gguf", "__init__.py"), "w").write("")
        os.makedirs(os.path.join(d, "up", "a", "b"))
        self.assertEqual(str(locate.gguf_py(os.path.join(d, "up", "a", "b"))), os.path.join(d, "up", "gguf-py"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
