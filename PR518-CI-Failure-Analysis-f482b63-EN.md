# PR #518 CI Failure Analysis Report (f482b63)

> Date: 2026-08-08
> PR: https://github.com/sgl-project/sgl-kernel-npu/pull/518
> Commit: `f482b636b1778db691c34a278c494d8b527d55b6`
> Branch: EdwardXuy/sgl-kernel-npu:Cover-kernel → main

---

## 1. Current Configuration

| File | CANN Version | Hardware | Mechanism |
|------|-------------|----------|-----------|
| daily-test-kernels.yml | [8.5.0, 9.0.0] matrix | [a3] | strategy.matrix, 2 jobs |
| pr-test-kernels.yml | 9.0.0 fixed | a3 | 7 independent jobs, no matrix |

> Note: The daily yml (lines 21-23) contains A2 (910b) conditional logic, but `hardware: [a3]` does not enable A2. A2 is "dead code."

---

## 2. CI Overview

| Workflow | Result | Note |
|----------|--------|------|
| Lint | ✅ pass | |
| Build and Release | ✅ pass | |
| A2 Internode Test | ✅ pass | |
| Release DeepEP Wheel | ✅ pass | |
| DeepEP PR Test | ❌ fail | **Upstream issue** — also fails on main |
| Daily Kernel (8.5.0, a3) | ❌ fail | 12 failures (triton module + test defects) |
| Daily Kernel (9.0.0, a3) | ❌ fail | 7 failures (6 test defects + 1 flaky) |
| PR Test (9.0.0) | ❌ fail | 4 jobs fail, 3 pass |

---

## 3. Daily 9.0.0 Results: Total 40, Failed 7

| # | Test | Error | Category | Developer Fix |
|---|------|-------|----------|---------------|
| 1 | test_add_rmsnorm_bias.py | `TypeError: add_rmsnorm_bias() got multiple values for argument 'norm_bias'` | API param change | Yes |
| 2 | test_decode_attention.py | `AttributeError: module 'triton.language' has no attribute 'parallel'` | triton API incompat | Yes |
| 3 | test_split_qkv_rmsnorm_rope.py | `TypeError: custom_rope() missing 'half_rope_dim'` | API param change | Yes |
| 4 | test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py | `ModuleNotFoundError: No module named 'sglang'` | Missing dependency | Yes |
| 5 | test_conv1d_prefill.py | `RuntimeError: INTERNAL ASSERT FAILED (function_schema.cpp:547)` | torch_npu compat | Yes |
| 6 | test_swiglu_quant.py | `NameError: name 'F' is not defined` | Missing import | Yes |
| 7 | test_catlass_matmul_basic.py | `AssertionError: 0.015625 > 0.0005` | **Flaky precision** | Yes |

> **test_catlass_matmul_basic.py is flaky**: e2ba7d8 reported 0.0078 (fail), 308c781 passed, f482b63 reported 0.0156 (fail). The precision value varies per run — non-deterministic floating-point error.

---

## 4. Daily 8.5.0 Results: Total 40, Failed 12

### 8.5.0-Only Failures (triton.language.extra.cann missing, 6 tests)

triton-ascend 3.2.0 does not provide `triton.language.extra.cann`, which the sgl_kernel_npu package imports at runtime:

| Test | Cause |
|------|-------|
| test_verify_tree.py | `ModuleNotFoundError: No module named 'triton.language.extra.cann'` |
| test_mamba_conv.py | same |
| test_gated_delta_ascendc_tri_inv.py | same |
| test_chunk_gdn_pto.py | same |
| test_chunk_gdn_triton.py | same |
| test_solve_tril.py | same |

### Failures Common to Both 8.5.0 and 9.0.0 (6 tests)

| Test | 8.5.0 Error | 9.0.0 Error |
|------|------------|------------|
| test_add_rmsnorm_bias.py | triton module missing | TypeError: norm_bias |
| test_decode_attention.py | fail | AttributeError: triton.language.parallel |
| test_split_qkv_rmsnorm_rope.py | triton module missing | TypeError: custom_rope |
| test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py | sglang missing | sglang missing |
| test_conv1d_prefill.py | RuntimeError | RuntimeError |
| test_swiglu_quant.py | NameError: F | NameError: F |

### 8.5.0 vs 9.0.0 Comparison

| | 8.5.0 | 9.0.0 |
|---|---|---|
| Total tests | 40 | 40 |
| Failures | **12** | **7** |
| triton module missing | 6 unique | 0 |
| Flaky precision | not triggered | 1 (catlass) |
| Test defects | 6 common | 6 common |

---

## 5. PR Test 9.0.0 Results

| Job | Result | Failed Tests |
|-----|--------|-------------|
| test-cache-ops | ✅ pass | — |
| test-fla-ops | ✅ pass | — |
| test-speculative-ops | ✅ pass | — |
| test-norm-ops | ❌ fail | add_rmsnorm_bias |
| test-attention-ops | ❌ fail | decode_attention, split_qkv_rmsnorm_rope, split_qkv_rmsnorm_rope_pos_cache_half_npu |
| test-fused-ops | ❌ fail | swiglu_quant |
| test-mamba-ops | ❌ fail | conv1d_prefill |

---

## 6. Failure Classification

### Category A: Upstream Issue (Unrelated to PR #518)
**DeepEP PR Test**: `pr-test-deepep-npu.yml` fails on main itself (`652edf6` #646, `9757e73` #660).

### Category B: CANN 8.5.0 Environment Issue (triton-ascend 3.2.0)
`triton.language.extra.cann` does not exist in 3.2.0, causing 6 extra failures vs 9.0.0. **Unrelated to submodules** (submodules fix compile-time PTO-ISA; cann extra is a runtime pip-package content issue).

### Category C: Genuine Test Defects (6 stable + 1 flaky, require developer fixes)

| Test | Fix Suggestion |
|------|---------------|
| test_add_rmsnorm_bias.py | Pass norm_bias as keyword argument |
| test_decode_attention.py | Adapt to triton-ascend 3.2.1 (no triton.language.parallel) |
| test_split_qkv_rmsnorm_rope.py | Pass new required arg half_rope_dim to custom_rope |
| test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py | try/except for sglang import |
| test_conv1d_prefill.py | Investigate torch_npu version compatibility |
| test_swiglu_quant.py | Add `import torch.nn.functional as F` |
| test_catlass_matmul_basic.py | Relax precision threshold or fix kernel (0.0005 too strict) |

---

## 7. Fix History

| Commit | Change | Effect |
|--------|--------|--------|
| `308c781` | merge main + CANN 9.0.0 + sync test list | Failures 7→6 (catlass temporarily passed), Lint/Internode fixed |
| `149e1e6` | restore daily [8.5.0,9.0.0] + PR test 8.5.0 | Verify 8.5.0 (12 failures) |
| `f482b63` | PR test back to 9.0.0 | PR test 3/7 pass, speculative fixed |

---

## 8. Next Steps

1. Report **6 stable test defects** to developers (table above)
2. **catlass flaky**: relax threshold or mark xfail
3. **CANN 8.5.0**: identify a triton-ascend version that supports 8.5.0 AND includes cann extra module
4. **A2 hardware**: daily yml has conditional logic; add `hardware: [a3, a2]` once tests stabilize
5. **DeepEP**: await upstream fix

---

## 9. Log Archive

| File | Description |
|------|-------------|
| `logs-f482b63-20260808-124347/f482-daily-850-log.txt` | Daily 8.5.0 (12 failures) |
| `logs-f482b63-20260808-124347/f482-daily-900-log.txt` | Daily 9.0.0 (7 failures) |
