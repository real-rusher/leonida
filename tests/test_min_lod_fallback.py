"""--min-lod-fallback in tools/rdna2spirv.py: image_sample*_cl without the MinLod image operand.

Intel's Windows iGPU drivers have no shaderResourceMinLod, so a module with OpCapability MinLod does not load there.
With the switch, the translator samples with explicit gradients instead and scales them so that the LOD the device
computes from them is at least the clamp (Gen.min_lod_grads). These tests need neither llvm-mc nor a GPU:

* the shape of the SPIR-V for every _cl form, with and without the switch (and with --texel-offset-fallback);
* the arithmetic of Gen.min_lod_grads, evaluated in binary32 over the ops it emits and checked against Vulkan's LOD
  formula ("Scale Factors and Level of Detail"): gradients whose LOD is at least the clamp pass bit for bit, the
  others come out at the clamp, with the S#'s anisotropy and LOD bias, literal or read at run time.

The sampled pixels are checked on a GPU by runtime/test/shaders/ps_texture_cl.asm, ps_texture_b_cl.asm and
ps_texture_grad_cl_a16.asm with LEONIDA_MINLOD_FALLBACK=1 (docs/testing.md).
"""
import math
import re
import shutil
import struct
import subprocess

import numpy as np
import pytest

from listing import translate_listing
import rdna2spirv  # noqa: E402  (put on sys.path by listing)

DESC = "s[8:15], s[16:19]"
PROLOGUE = "\n".join(f"v_mov_b32 v{r}, 0" for r in [3] + list(range(10, 20)))
# (instruction, SPIR-V sample op without the switch, has a bias, has explicit gradients)
FORMS = [
    (f"image_sample_cl v[4:7], v[10:12], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D", "OpImageSampleImplicitLod", False, False),
    (f"image_sample_b_cl v[4:7], v[10:13], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D", "OpImageSampleImplicitLod", True, False),
    (f"image_sample_c_cl v4, v[10:13], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D", "OpImageSampleDrefImplicitLod", False, False),
    (f"image_sample_c_b_cl v4, v[10:14], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D", "OpImageSampleDrefImplicitLod", True, False),
    (f"image_sample_d_cl v[4:7], v[10:16], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D", "OpImageSampleExplicitLod", False, True),
    (f"image_sample_d_cl v[4:7], v[10:13], {DESC} dmask:0xf dim:SQ_RSRC_IMG_1D", "OpImageSampleExplicitLod", False, True),
    (f"image_sample_d_cl v[4:7], v[10:19], {DESC} dmask:0xf dim:SQ_RSRC_IMG_3D", "OpImageSampleExplicitLod", False, True),
    (f"image_sample_d_cl v[4:7], v[10:17], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY", "OpImageSampleExplicitLod", False, True),
    (f"image_sample_cl v[4:7], v[10:13], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY", "OpImageSampleImplicitLod", False, False),
    (f"image_sample_cl v[4:7], v[10:13], {DESC} dmask:0xf dim:SQ_RSRC_IMG_3D", "OpImageSampleImplicitLod", False, False),
    (f"image_sample_cl v[4:7], v[10:11], {DESC} dmask:0xf dim:SQ_RSRC_IMG_1D", "OpImageSampleImplicitLod", False, False),
    (f"image_sample_cl v[4:7], v[10:13], {DESC} dmask:0xf dim:SQ_RSRC_IMG_CUBE", "OpImageSampleImplicitLod", False, False),
    (f"image_sample_cl_o v[4:7], v[10:13], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D", "OpImageSampleImplicitLod", False, False),
    (f"image_sample_c_b_cl_o v4, v[10:15], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D", "OpImageSampleDrefImplicitLod", True, False),
]
IDS = [f[0].split()[0] + "_" + f[0].split("SQ_RSRC_IMG_")[1] for f in FORMS]


@pytest.fixture
def fallback(monkeypatch):
    monkeypatch.setattr(rdna2spirv, "MIN_LOD_FALLBACK", True)


def _samples(spv):
    return [ln for ln in spv.splitlines() if re.search(r"= OpImageSample\w+ ", ln)]


@pytest.mark.parametrize("asm,op,bias,grad", FORMS, ids=IDS)
def test_without_the_switch_the_clamp_is_min_lod(asm, op, bias, grad, tmp_path):
    spv = translate_listing(tmp_path, PROLOGUE + "\n" + asm)
    samples = _samples(spv)
    assert samples and all(f"= {op} " in ln and "MinLod" in ln for ln in samples), samples
    assert "OpCapability MinLod" in spv


@pytest.mark.parametrize("asm,op,bias,grad", FORMS, ids=IDS)
def test_fallback_samples_with_gradients_and_no_min_lod(asm, op, bias, grad, tmp_path, fallback):
    spv = translate_listing(tmp_path, PROLOGUE + "\n" + asm)
    samples = _samples(spv)
    explicit = "OpImageSampleDrefExplicitLod" if "Dref" in op else "OpImageSampleExplicitLod"
    assert samples and all(f"= {explicit} " in ln for ln in samples), samples
    assert not any("MinLod" in ln for ln in samples), samples
    assert all(re.search(r" Grad(\|Offset)? %\w+ %\w+", ln) for ln in samples), samples
    assert "OpCapability MinLod" not in spv
    assert "OpImageQuerySizeLod" in spv and "OpCapability ImageQuery" in spv
    # implicit forms take the derivatives of the coordinate (x 2^bias for the _b forms)
    assert ("OpDPdx" in spv and "OpDPdy" in spv) == (not grad)
    assert ("%glsl Exp2" in spv)            # 2^(clamp - sampler bias) is always computed
    if bias:
        assert len(re.findall(r"%glsl Exp2", spv)) >= 2, "the shader's bias scales the derivatives"
    if "_o " in asm:   # the texel offset stays an image operand next to the gradients
        assert all("Grad|Offset" in ln for ln in samples), samples


@pytest.mark.parametrize("asm,op,bias,grad", [f for f in FORMS if "_o " in f[0]], ids=[i for i in IDS if "_o_" in i])
def test_fallback_with_texel_offset_fallback(asm, op, bias, grad, tmp_path, fallback, monkeypatch):
    """Both switches (an Intel iGPU has neither feature): the folded offset's gradients are the ones scaled."""
    monkeypatch.setattr(rdna2spirv, "TEXEL_OFFSET_FALLBACK", True)
    spv = translate_listing(tmp_path, PROLOGUE + "\n" + asm)
    samples = _samples(spv)
    assert samples and not any("MinLod" in ln or "Offset" in ln for ln in samples), samples
    assert "OpCapability MinLod" not in spv
    assert len(re.findall(r"= OpImageQuerySizeLod ", spv)) >= 2   # the offset's texel size and the base level's


@pytest.mark.parametrize("asm", [f"image_sample v[4:7], v[10:11], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D",
                                 f"image_sample_d v[4:7], v[10:15], {DESC} dmask:0xf dim:SQ_RSRC_IMG_2D",
                                 f"image_sample_c_b v4, v[10:13], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D"],
                         ids=["sample", "sample_d", "sample_c_b"])
def test_fallback_leaves_samples_without_clamp_alone(asm, tmp_path, monkeypatch):
    before = translate_listing(tmp_path, PROLOGUE + "\n" + asm)
    monkeypatch.setattr(rdna2spirv, "MIN_LOD_FALLBACK", True)
    assert translate_listing(tmp_path, PROLOGUE + "\n" + asm) == before


@pytest.mark.skipif(not (shutil.which("spirv-as") and shutil.which("spirv-val")), reason="needs spirv-as and spirv-val")
@pytest.mark.parametrize("asm,op,bias,grad", FORMS, ids=IDS)
def test_fallback_modules_validate(asm, op, bias, grad, tmp_path, fallback):
    spv = translate_listing(tmp_path, PROLOGUE + "\n" + asm)
    src, out = tmp_path / "m.spvasm", tmp_path / "m.spv"
    src.write_text(spv, encoding="utf-8")
    for cmd in (["spirv-as", "--target-env", "vulkan1.2", str(src), "-o", str(out)],
                ["spirv-val", "--target-env", "vulkan1.2", str(out)]
                + (["--allow-offset-texture-operand"] if "_o " in asm else [])):   # the offset, not the clamp
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# ---- the arithmetic of Gen.min_lod_grads ----

f32 = np.float32


def _run(size, gx, gy, minlod, word0, word2, literal, ncoord=None):
    """Evaluate Gen.min_lod_grads in binary32 and return the new (gx, gy) as float32 arrays.

    size: the base level's size in the spatial dimensions; ncoord > len(size) adds an array layer to the queried
    size. word0 / word2: the S#'s words 0 and 2, literal (Gen.sgpr_imm) or loaded from the SGPRs at run time."""
    spatial = len(size)
    ncoord = ncoord or spatial
    g = rdna2spirv.Gen("ps")
    if literal:
        g.sgpr_imm = {16: word0, 18: word2}
    ox, oy = g.min_lod_grads("%img", ("%gx", "%gy"), ncoord, spatial, "%minlod", {"r": 16})
    consts = {name: v for v, name in g.consts.items()}
    vals = {"%gx": np.array(gx, f32).reshape(-1), "%gy": np.array(gy, f32).reshape(-1), "%minlod": f32(minlod)}
    chains = {}
    sgpr = {16: word0, 18: word2}

    def v(x):
        return consts[x] if x in consts else vals[x]

    def bits_f(u):
        return np.array([u & 0xFFFFFFFF], np.uint32).view(f32)[0]

    for line in g.body:
        m = re.match(r"\s*(%\w+) = (\w+) (\S+) ?(.*)$", line)
        dst, op, ty, rest = m[1], m[2], m[3], m[4].split()
        if op == "OpAccessChain":
            assert rest[0] == "%sgpr", line
            chains[dst] = consts[rest[1]]
            continue
        if op == "OpLoad":
            r = sgpr[chains[rest[0]]]
        elif op == "OpBitcast":
            x = v(rest[0])
            if ty == "%float":
                r = bits_f(int(x))
            elif ty == "%int":
                r = int(np.array([int(x) & 0xFFFFFFFF], np.uint32).view(np.int32)[0])
            else:
                r = int(x) & 0xFFFFFFFF
        elif op == "OpImageQuerySizeLod":
            assert rest[0] == "%img" and v(rest[1]) == 0, "the footprint is measured in texels of the base level"
            q = list(size) + [6] * (ncoord - spatial)
            r = np.array(q, np.int32)
        elif op == "OpVectorShuffle":
            r = v(rest[0])[[int(i) for i in rest[2:]]]
        elif op in ("OpConvertSToF", "OpConvertUToF"):
            r = np.array(v(rest[0]), f32).reshape(-1) if ty.startswith("%v") else f32(v(rest[0]))
            if ty == "%float":
                r = f32(np.array(v(rest[0])).reshape(-1)[0])
        elif op == "OpBitFieldUExtract":
            r = (v(rest[0]) >> v(rest[1])) & ((1 << v(rest[2])) - 1)
        elif op == "OpBitFieldSExtract":
            w = v(rest[2])
            f = (v(rest[0]) >> v(rest[1])) & ((1 << w) - 1)
            r = ((f ^ (1 << (w - 1))) - (1 << (w - 1))) & 0xFFFFFFFF
        elif op == "OpShiftLeftLogical":
            r = (v(rest[0]) << v(rest[1])) & 0xFFFFFFFF
        elif op in ("OpFMul", "OpVectorTimesScalar", "OpFDiv", "OpFAdd", "OpFSub"):
            a, b = v(rest[0]), v(rest[1])
            with np.errstate(all="ignore"):
                r = {"OpFMul": np.multiply, "OpVectorTimesScalar": np.multiply, "OpFDiv": np.divide,
                     "OpFAdd": np.add, "OpFSub": np.subtract}[op](np.asarray(a, f32), np.asarray(b, f32))
        elif op == "OpFOrdLessThan":
            r = bool(v(rest[0]) < v(rest[1]))
        elif op == "OpFOrdGreaterThan":
            r = bool(v(rest[0]) > v(rest[1]))
        elif op == "OpSelect":
            r = v(rest[1]) if v(rest[0]) else v(rest[2])
        elif op == "OpCompositeConstruct":
            r = np.array([v(c) for c in rest], f32)
        elif op == "OpExtInst":
            fn, args = rest[1], [v(c) for c in rest[2:]]
            with np.errstate(all="ignore"):
                if fn == "Length":
                    r = f32(np.sqrt(np.sum(np.square(np.asarray(args[0], f32)), dtype=f32)))
                elif fn == "FAbs":
                    r = np.abs(np.asarray(args[0], f32))
                elif fn == "FMax":
                    r = np.fmax(f32(args[0]), f32(args[1]))
                elif fn == "FMin":
                    r = np.fmin(f32(args[0]), f32(args[1]))
                elif fn == "UMin":
                    r = min(args[0], args[1])
                elif fn == "Log2":
                    r = np.log2(f32(args[0]))
                elif fn == "Exp2":
                    r = np.exp2(f32(args[0]))
                else:
                    raise AssertionError(line)
        else:
            raise AssertionError(f"op not evaluated: {line}")
        if isinstance(r, np.ndarray) and r.shape == (1,) and not ty.startswith("%v"):
            r = r[0]
        vals[dst] = r
    return np.asarray(vals[ox], f32).reshape(-1), np.asarray(vals[oy], f32).reshape(-1)


def _lod(size, gx, gy, word0, word2):
    """Vulkan's lambda (before the clamp) in float64: log2(rho_max / eta) + the sampler's bias."""
    s = np.array(size, float)
    rx = np.linalg.norm(np.asarray(gx, float) * s)
    ry = np.linalg.norm(np.asarray(gy, float) * s)
    rmax, rmin = max(rx, ry), min(rx, ry)
    aniso = float(1 << min(word0 >> 9 & 7, 4))
    eta = min(rmax / rmin, aniso) if rmin > 0 else aniso
    b14 = word2 & 0x3FFF
    return math.log2(rmax / eta) + ((b14 ^ 0x2000) - 0x2000) / 256.0


def _word0(aniso_field):
    return 0x92 | aniso_field << 9


def _word2(bias):
    return 0x14500000 | (round(bias * 256) & 0x3FFF)


CASES = [   # (size, gx, gy, minlod, aniso field, sampler bias)
    ((64, 64), (1 / 64, 0), (0, 1 / 64), 0.0, 0, 0.0),           # LOD 0, clamp 0: unchanged
    ((64, 64), (1 / 64, 0), (0, 1 / 64), 1.0, 0, 0.0),           # LOD 0, clamp 1: raised to 1
    ((64, 64), (1 / 64, 0), (0, 1 / 64), 2.75, 0, 0.0),          # a fractional clamp
    ((64, 64), (4 / 64, 0), (0, 4 / 64), 1.0, 0, 0.0),           # LOD 2 above the clamp: unchanged
    ((256, 64), (0.3 / 256, 0.1 / 64), (-0.2 / 256, 0.7 / 64), 3.0, 0, 0.0),   # skewed gradients, non-square
    ((64, 64), (8 / 64, 0), (0, 1 / 64), 2.0, 0, 0.0),           # anisotropic, no anisotropic filtering
    ((64, 64), (8 / 64, 0), (0, 1 / 64), 2.0, 1, 0.0),           # max anisotropy 2: LOD log2(8 / 2) = 2
    ((64, 64), (8 / 64, 0), (0, 1 / 64), 3.0, 2, 0.0),           # 4x: LOD 1, clamp 3
    ((64, 64), (32 / 64, 0), (0, 1 / 64), 3.0, 7, 0.0),          # field 7 is capped at 16x
    ((64, 64), (1 / 64, 0), (0, 1 / 64), 1.0, 0, -1.5),          # negative sampler bias
    ((64, 64), (1 / 64, 0), (0, 1 / 64), 1.0, 0, 0.5),           # positive bias below the clamp
    ((64, 64), (1 / 64, 0), (0, 1 / 64), 1.0, 0, 2.0),           # bias alone above the clamp: unchanged
    ((64, 64), (1 / 64, 0), (0, 1 / 64), 13.0, 0, 0.0),          # a clamp past the last level
    ((128,), (1 / 128,), (0.5 / 128,), 2.0, 0, 0.0),             # 1D
    ((32, 32, 16), (1 / 32, 0, 0), (0, 0, 1 / 16), 1.5, 0, 0.0),  # 3D
    ((32, 32, 16), (2 / 32, 1 / 32, 3 / 16), (0, 1 / 32, 0), 0.5, 2, -0.25),   # 3D, unchanged
]


@pytest.mark.parametrize("literal", [True, False], ids=["s_imm", "s_runtime"])
@pytest.mark.parametrize("size,gx,gy,minlod,aniso,sbias", CASES)
def test_min_lod_grads_give_the_clamped_lod(size, gx, gy, minlod, aniso, sbias, literal):
    w0, w2 = _word0(aniso), _word2(sbias)
    nx, ny = _run(size, gx, gy, minlod, w0, w2, literal)
    lam = _lod(size, gx, gy, w0, w2)
    if lam >= minlod + 1e-6:
        # bit for bit: x 1.0
        assert nx.tobytes() == np.array(gx, f32).tobytes() and ny.tobytes() == np.array(gy, f32).tobytes()
    else:
        assert _lod(size, nx, ny, w0, w2) == pytest.approx(minlod, abs=1e-4)
        # the scale is common to both gradients: their directions and ratio (and so eta) are kept
        k = float(np.linalg.norm(nx)) / float(np.linalg.norm(np.array(gx, float)))
        assert np.allclose(ny, np.array(gy, float) * k, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("ncoord", [3], ids=["array_or_cube"])
def test_min_lod_grads_query_only_the_spatial_size(ncoord):
    """A 2D array (and a cube, bound as a 6-layer 2D array) has a layer count in the size query; it is not a
    dimension of the texel footprint."""
    w0, w2 = _word0(0), _word2(0.0)
    nx, ny = _run((64, 32), (1 / 64, 0), (0, 1 / 32), 2.0, w0, w2, True, ncoord=ncoord)
    assert _lod((64, 32), nx, ny, w0, w2) == pytest.approx(2.0, abs=1e-4)


@pytest.mark.parametrize("bad", [0.0, 1e-35, float("nan")], ids=["zero", "denormal_scale", "nan"])
@pytest.mark.parametrize("sbias", [0.0, -1.0])
def test_min_lod_grads_replace_degenerate_gradients(bad, sbias):
    """Zero, NaN or tiny gradients: an isotropic footprint of 2^(clamp - bias) texels, so LOD = clamp."""
    size, minlod = (64, 16), 1.5
    w0, w2 = _word0(2), _word2(sbias)
    nx, ny = _run(size, (bad, bad), (bad, bad), minlod, w0, w2, True)
    assert np.all(np.isfinite(nx)) and np.all(np.isfinite(ny))
    texels = 2.0 ** (minlod - sbias)
    assert np.allclose(nx * np.array(size, f32), [texels, 0], rtol=1e-6)
    assert np.allclose(ny * np.array(size, f32), [0, texels], rtol=1e-6)
    assert _lod(size, nx, ny, w0, w2) == pytest.approx(minlod, abs=1e-5)


def test_min_lod_grads_extreme_clamp_stays_finite():
    """The exponent is held at 64, so a clamp from a stray register can't turn the gradients into inf."""
    nx, ny = _run((64, 64), (1 / 64, 0), (0, 1 / 64), 1e30, _word0(0), _word2(0.0), True)
    assert np.all(np.isfinite(nx)) and np.all(np.isfinite(ny))
    assert _lod((64, 64), nx, ny, _word0(0), _word2(0.0)) == pytest.approx(64.0, abs=1e-3)


def test_min_lod_grads_literal_and_runtime_s_words_agree():
    for aniso in range(8):
        for sbias in (-16.0, -3.25, 0.0, 0.00390625, 7.5, 15.99609375):
            args = ((64, 64), (5 / 64, 1 / 64), (-1 / 64, 2 / 64), 4.0, _word0(aniso), _word2(sbias))
            a, b = _run(*args, literal=True), _run(*args, literal=False)
            assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes(), (aniso, sbias)
