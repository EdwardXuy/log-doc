# PR #518 CI Failure Analysis Report (2e39708)

> Date: 2026-08-12
> PR: https://github.com/sgl-project/sgl-kernel-npu/pull/518
> Commit: `2e39708` (latest, CI results now available)
> Previous commit: `2245298` (force-reinstall fix)
> Branch: EdwardXuy/sgl-kernel-npu:Cover-kernel → main

---

## 1. Context

This report analyzes the CI status of PR #518 at commit `2e39708`. The CI results for `2e39708` are now available, allowing a definitive assessment of the two fixes applied in this commit (the `tl.parallel` compatibility shim and the `ACL_OP_SELECT_IMPL_MODE=high_precision` switch). This update supersedes the prior speculative analysis.

| Item | Value |
|------|-------|
| PR | https://github.com/sgl-project/sgl-kernel-npu/pull/518 |
| Latest commit | `2e39708` |
| Previous commit | `2245298` (force-reinstall fix) |
| Branch | `EdwardXuy/sgl-kernel-npu:Cover-kernel` → `main` |
| Date | 2026-08-12 |

---

## 2. CI Results (commit `2e39708`)

### 2.1 PR Test: 3 passed / 4 failed (5 test failures total)

| Job | Result | Failures |
|-----|--------|----------|
| test-cache-ops | ✅ pass | 0 |
| test-speculative-ops | ✅ pass | 0 |
| test-fla-ops | ✅ pass | 0 |
| test-norm-ops | ❌ fail | 1: test_add_rmsnorm_bias |
| test-attention-ops | ❌ fail | 2: test_decode_attention, test_split_qkv_rmsnorm_rope |
| test-mamba-ops | ❌ fail | 1: test_conv1d_prefill |
| test-fused-ops | ❌ fail | 1: test_swiglu_quant |

### 2.2 Daily CANN 9.0.0: 6 failures (out of 40 tests, 34 passed)

1. test_add_rmsnorm_bias
2. test_decode_attention
3. test_split_qkv_rmsnorm_rope
4. test_conv1d_prefill
5. test_swiglu_quant
6. test_catlass_matmul_basic (flaky — only in Daily, not in PR test)

### 2.3 Daily CANN 8.5.0: large number of failures

triton-ascend 3.2.0 is incompatible with the CANN 8.5.0 stack (missing `triton.language.extra.cann` module). This leg is expected to fail and is not driven by PR #518.

---

## 3. Why Daily Has More Failures Than PR Test (6 vs 5)

The difference is **test_catlass_matmul_basic.py** — a **flaky precision test**:

- It uses random shapes for testing.
- In Daily, the random shape (427, 195, 366) caused `AssertionError: 0.0078125 not less than or equal to 0.0005`.
- In PR test, the random shape happened to produce a smaller error and passed.
- This is NOT a systematic failure — it's a float16 precision fluctuation.

PR test runs 39 tests (7 groups, no SMOKE_TESTS). Daily runs 40 tests (ALL_TESTS includes `test_hello_world.py`). The 1-test scope difference (`test_hello_world.py`) doesn't affect failures since it passes.

---

## 4. Triton-Ascend Installation

triton-ascend is **separately installed** in `prepare_kernel_tests.sh`, NOT from the docker image:

```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

The docker image may have triton-ascend pre-installed, but our script overwrites it with the specified version.

---

## 5. Commit `2e39708` Workaround Effectiveness

### 5.1 tl.parallel compatibility shim — PARTIALLY WORKED

- **Before**: `AttributeError: module 'triton.language' has no attribute 'parallel'`
- **After**: `RuntimeError: Only range and static_range iterators are currently supported`
- **Analysis**: The shim successfully added the `parallel` attribute to `triton.language`, so `getattr()` no longer fails. However, the triton JIT compiler checks loop iterators at the AST level and only accepts `range` and `static_range` — it doesn't accept the return value of `tl.parallel()` even though it returns a Python `range` object.
- **Conclusion**: Cannot be fixed via CI shim. The source code `decode_attention.py` line 533 needs to be updated to use `tl.range()` instead of `tl.parallel()`.

### 5.2 ACL_OP_SELECT_IMPL_MODE=high_precision — DID NOT HELP

- `test_swiglu_quant` still fails with `assert max_diff <= 1` (AssertionError).
- The `high_precision` mode doesn't affect the custom kernel's quantization precision.

---

## 6. Failure Analysis (6 failures in Daily 9.0.0)

### Category A: Source code issue — tl.parallel removed (1 failure)

- **test_decode_attention.py**: `tl.parallel(0, 2, bind_sub_block=True)` in `decode_attention.py:533`
- triton 3.5.0 removed `tl.parallel`; only supports `range` and `static_range`.
- CI shim partially worked but JIT compiler rejects it.
- **Fix**: Modify source code to use `tl.range()` or `tl.static_range()`.

### Category B: Test code bug — missing parameter (1 failure)

- **test_split_qkv_rmsnorm_rope.py**: `custom_rope(_q, _k, sin, cos)` at line 165
- Function definition at line 10 requires `half_rope_dim` parameter.
- **Fix**: Add `half_rope_dim` argument at line 165.

### Category C: API parameter mismatch (1 failure)

- **test_add_rmsnorm_bias.py**: `TypeError: add_rmsnorm_bias() got multiple values for argument 'norm_bias'`
- **Fix**: Modify test code parameter passing.

### Category D: PyTorch 2.10.0 strictness (1 failure)

- **test_conv1d_prefill.py**: `RuntimeError: !argument.default_value() INTERNAL ASSERT FAILED`
- torch 2.10.0 enforces stricter function schema validation.
- **Fix**: Add default values in operator registration, or use torch 2.8.0.

### Category E: Precision issues (2 failures)

- **test_swiglu_quant.py**: `assert max_diff <= 1` — quantization precision exceeds tolerance.
- **test_catlass_matmul_basic.py**: `0.0078125 > 0.0005` — float16 random shape precision (FLAKY).
- **Fix**: Adjust tolerance thresholds or improve algorithm precision.

### Category F: CANN 8.5.0 incompatibility (Daily only)

- `ModuleNotFoundError: No module named 'triton.language.extra.cann'`
- triton-ascend 3.2.0 lacks critical module.
- **Fix**: None available from CI side.

---

## 7. Fixes Applied (commit history)

### 7.1 Commit `2245298` — force-reinstall (EFFECTIVE)

- Fixed 3 failures: test_catlass_matmul_basic (systematic), test_gmm_wfp8a16, test_mm_wfp8a16.
- Docker image pre-installed `sgl_kernel_npu==2026.6.1`; pip skipped installation.
- Fix: `--force-reinstall --no-deps`.

### 7.2 Commit `2e39708` — tl.parallel shim + high_precision (LIMITED EFFECT)

- **tl.parallel shim**: Changed error from `AttributeError` to `RuntimeError` (progress but not fixed).
- **high_precision mode**: No effect on `test_swiglu_quant`.

---

## 8. Remaining Failures (cannot fix from CI side)

1. **test_decode_attention** — source code needs `tl.parallel` → `tl.range` replacement.
2. **test_split_qkv_rmsnorm_rope** — test code bug (missing parameter).
3. **test_add_rmsnorm_bias** — API parameter mismatch.
4. **test_conv1d_prefill** — torch 2.10.0 strictness.
5. **test_swiglu_quant** — quantization precision.
6. **test_catlass_matmul_basic** — flaky float16 precision (may pass on rerun).

---

## 9. Recommendations

1. Report `decode_attention.py:533` `tl.parallel` issue to developers — needs source code fix.
2. Report `test_split_qkv_rmsnorm_rope.py:165` missing parameter bug to developers.
3. Report `test_add_rmsnorm_bias.py` API mismatch to developers.
4. Consider torch 2.8.0 (stable image) for `test_conv1d_prefill`.
5. CANN 8.5.0 daily: add `continue-on-error: true`.
6. `test_catlass_matmul_basic`: may pass on rerun (flaky).

---

## 10. Summary

| Category | Count | Owner | Status after `2e39708` (observed) |
|----------|-------|-------|-----------------------------------|
| A. Source code (`tl.parallel` removed) | 1 test | Developer | Shim partially worked; still failing — needs source fix |
| B. Test code bug (`half_rope_dim`) | 1 test | Developer | Still failing |
| C. API parameter mismatch | 1 test | Developer | Still failing |
| D. PyTorch 2.10.0 strictness | 1 test | Developer / CI | Still failing |
| E. Precision (swiglu_quant + flaky catlass) | 2 tests | Developer | `high_precision` had no effect; catlass is flaky |
| F. CANN 8.5.0 incompatibility | daily only | Upstream | Expected failure |

The `2e39708` commit's two workarounds had limited effect: the `tl.parallel` shim advanced the error from `AttributeError` to `RuntimeError` (progress, but not a fix), and `ACL_OP_SELECT_IMPL_MODE=high_precision` had no effect on `test_swiglu_quant`. The remaining 6 failures require developer-side source/test fixes, a torch version decision, or upstream triton-ascend compatibility work. The `test_catlass_matmul_basic` failure is flaky and may pass on a rerun.
