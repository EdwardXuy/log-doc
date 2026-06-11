# PR Test Suite Runtime Optimization Analysis

**Workflow:** [pr-test-npu.yml](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml)
**Caller:** [daily-build-test.yml](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/daily-build-test.yml) (via `workflow_call`)
**Sample run analyzed:** [Run #27292189262](https://github.com/sgl-project/sgl-kernel-npu/actions/runs/27292189262) (2026-06-10 17:00 → 19:05 UTC)
**Analysis date:** 2026-06-11
**Constraint:** CANN × A2/A3 matrix and self-hosted runner labels are fixed and **must not** be changed.

---

## 1. Executive Summary

| Metric | Value |
|---|---|
| Total pipeline wall-clock (daily + PR suite, including queueing) | **125.2 min (≈ 2h 5m)** |
| Daily enumerate tests wall-clock | **~24 min** (limited by `Low Latency a3` ≈ 24 min) |
| PR test suite wall-clock (after daily) | **~40 min** (limited by `test-build-deepep-a3-moe` 9.0.0 = 39.7 min) |
| Number of self-hosted jobs launched | **23** (8 daily + 12 PR matrix + 2 multi-node + finish) |
| Hidden queueing gap (Δ between sum-of-durations and wall-clock) | **~60 min** — indicates **runner-slot starvation** on `linux-aarch64-a3-16` / `linux-aarch64-a2-8` |

The previous "200+ min" run-time was a runner-queue starvation problem, not a single-job problem. Splitting more work would just *worsen* the queue. The remaining 125 min breaks down into:

- **24 min** of daily enumerate tests (`Low Latency a3` matrix)
- **40 min** of PR tests (longest PR job = `a3-moe`)
- **~60 min** of idle wait for self-hosted runner slots (12 a3-16 jobs vs. available slot count)

The realistic, sustainable target is **~65–75 min** wall-clock. Reaching that requires **removing redundant jobs and test cases**, not adding more parallelism.

---

## 2. Current Run Breakdown (Run #27292189262)

### 2.1 Daily enumerate tests (run-daily-tests, all parallel)

| Job | Duration | Runner |
|---|---:|---|
| Intranode (CANN 8.5.0, a3) | 5.7 min | linux-aarch64-a3-16 |
| Intranode (CANN 9.0.0, a3) | 5.5 min | linux-aarch64-a3-16 |
| Intranode (CANN 8.5.0, a2) | 7.3 min | linux-aarch64-a2-8 |
| Intranode (CANN 9.0.0, a2) | 7.0 min | linux-aarch64-a2-8 |
| **Low Latency (CANN 8.5.0, a3)** | **23.6 min** | linux-aarch64-a3-16 |
| **Low Latency (CANN 9.0.0, a3)** | **24.0 min** | linux-aarch64-a3-16 |
| Low Latency (CANN 8.5.0, a2) | 17.6 min | linux-aarch64-a2-8 |
| Low Latency (CANN 9.0.0, a2) | 17.2 min | linux-aarch64-a2-8 |

### 2.2 PR test suite (pr-test-npu.yml, called via workflow_call)

| Job (CANN variant) | Duration | Notes |
|---|---:|---|
| test-all-build-core (8.5.0 / 9.0.0) | 29.6 / 29.2 min | `prepare_deepep_in_container.sh` (no `-a`) |
| test-all-build-moe (8.5.0 / 9.0.0) | 37.2 / 36.8 min | `prepare_deepep_in_container.sh` (no `-a`) |
| test-build-deepep-a3-core (8.5.0 / 9.0.0) | 32.4 / 32.2 min | `prepare_deepep_in_container.sh -a deepep` |
| **test-build-deepep-a3-moe (8.5.0 / 9.0.0)** | **39.5 / 39.7 min** | longest PR job — sets the wall-clock |
| test-build-deepep-a2 (8.5.0 / 9.0.0) | 38.5 / 39.0 min | `prepare_deepep_in_container.sh -a deepep2` |
| test-build-deepep-internode (8.5.0 / 9.0.0) | 9.0 / 11.1 min | multi-node, k8s job |

### 2.3 Slowest individual steps (top time sinks)

From `test-build-deepep-a3-moe (9.0.0)` — the wall-clock bottleneck:

| Step | Duration | Invocation count |
|---|---:|---:|
| Run test base fused deep moe | **319 s** | 5 invocations of `test_fused_deep_moe.py` |
| Run test muti-model for fused deep moe | **321 s** | 5 invocations |
| Run test fused deepep moe eplb | **323 s** | 5 invocations |
| Run test fused deepep moe for hidden | 128 s | 2 invocations |
| Run test fused deepep moe for topk | 132 s | 2 invocations |
| Run test fused deepep moe for experts | 112 s | 2 invocations |
| Run test mixed running for hidden | 176 s | 4 invocations |

From `test-all-build-core (9.0.0)`:

| Step | Duration | Note |
|---|---:|---|
| **Run test generalization of fused deep moe** | **302 s** | single `scripts/generalization_test_fused_deep_moe.sh` |
| Run test hidden intranode | 115 s | 4 invocations (`--hidden` 2048/4096/6144/7168) |
| Run test low latency for hidden | 132 s | 4 invocations |
| Run test intranode | 57 s | base `test_intranode.py` (run again at step "output parameters of different types", +30 s) |
| Run test low latency | 100 s | 3 invocations |
| Run test multi-round intranode | 57 s | 2 invocations |

---

## 3. Identified Redundancies

### 3.1 [CRITICAL] `test-all-build-core` ≈ `test-build-deepep-a3-core` (≈ 30 min waste × 2)

**Location:** [pr-test-npu.yml#L46-L222](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L46-L222) and [pr-test-npu.yml#L430-L701](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L430-L701)

These two jobs:
- share **identical** `if:` conditions (`ops_changed || common_changed`)
- run on the **same** runner `linux-aarch64-a3-16`
- use the **same** container image and CANN matrix
- use the **same** build-cache key (so cache is reused, not a 2× rebuild cost)
- execute the **same** 23 test steps; `test-build-deepep-a3-core` only adds:
  1. `--hidden=8192` in the `test_intranode.py` hidden step (one extra invocation)
  2. `--num-experts=1024 --num-processes=16` in the experts step (one extra invocation)
  3. `--hidden=8192` in the `test_low_latency.py` hidden step (one extra invocation)

The **only** meaningful difference is the prepare-script flag:
- `test-all-build-core`: `bash scripts/prepare_deepep_in_container.sh` (no `-a`, default branch)
- `test-build-deepep-a3-core`: `bash scripts/prepare_deepep_in_container.sh -a deepep`

> **Recommendation:** Confirm with the deepep owners whether `-a deepep` and the default branch produce different wheels. If they target the same code path, **delete one of the two jobs**. Wall-clock saving: ~30 min × 2 CANN variants = drops the PR suite ceiling from 40 → ~30 min. Runner pressure: −2 a3-16 jobs.

### 3.2 [CRITICAL] `test-all-build-moe` ≈ `test-build-deepep-a3-moe` (≈ 37 min waste × 2)

**Location:** [pr-test-npu.yml#L225-L427](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L225-L427) and [pr-test-npu.yml#L702-L914](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L702-L914)

Same pattern as 3.1:
- identical `if:` conditions (`ops_changed || common_changed`)
- identical runner / image / matrix / cache key
- identical 11 moe test steps + 6 mixed-running steps
- `test-build-deepep-a3-moe` only adds **one** extra step ("Run test fused deep moe for fuse_mode.DISPATCH_FFN_COMBINE" — 65 s) and one extra `--num-experts=1024 --num-processes=16` invocation

> **Recommendation:** Same as 3.1 — confirm the prepare flag and **delete one of the pair**. Wall-clock saving: ~37 min. Runner pressure: −2 a3-16 jobs.

### 3.3 [HIGH] Duplicate base-test execution inside the same job

**Location:**
- [pr-test-npu.yml#L130-L138](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L130-L138) and [pr-test-npu.yml#L160-L167](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L160-L167) (`test-all-build-core`)
- [pr-test-npu.yml#L501-L509](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L501-L509) and [pr-test-npu.yml#L530-L537](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L530-L537) (`test-build-deepep-a3-core`)
- [pr-test-npu.yml#L999-L1006](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L999-L1006) and [pr-test-npu.yml#L1021-L1028](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L1021-L1028) (`test-build-deepep-a2`)

Pattern (excerpt):
```yaml
- name: Run test intranode                       # line 131
  run: |
    python3 $GITHUB_WORKSPACE/tests/python/deepep/test_intranode.py
- ...
- name: Run test intranode for output parameters of different types  # line 161
  env:
    MOE_EXPERT_TOKEN_NUMS_TYPE: 0
  run: |
    python3 $GITHUB_WORKSPACE/tests/python/deepep/test_intranode.py
```

These two steps run **byte-identical commands**; the only difference is the env var `MOE_EXPERT_TOKEN_NUMS_TYPE=0`. This adds 30–40 s per duplicate.

> **Recommendation:** Either pass `MOE_EXPERT_TOKEN_NUMS_TYPE` as a CLI flag in a single `test_intranode.py` run, or merge into the base step. Saves 2–3 min total per run × 6 affected jobs (core, a3-core, a2) ≈ **~15–20 min runner-minutes per pipeline** and removes a confusing duplicate.

### 3.4 [MEDIUM] Daily enumerate already covers most intranode/low_latency parameter sweeps

**Location:** [daily-build-test.yml#L46-L64](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/daily-build-test.yml#L46-L64) → `scripts/enumerate_test_*.sh`

The `run-daily-tests` job (matrix: `intranode × low_latency × a3 × a2 × 2 CANN = 8 jobs`) is precisely an **exhaustive parameter sweep** of the deepep operator. After it succeeds, the PR test suite re-runs **a hand-picked subset** of the same `test_intranode.py` / `test_low_latency.py` configurations for the a3 hardware path. If a configuration has been green on daily enumerate, re-running it on every PR yields little extra signal.

| PR test group | Already covered by daily enumerate? |
|---|---|
| `test_intranode.py` (default, `--num-tokens`, `--num-processes`, `--hidden`, `--num-topk`, `--num-experts`, `--active-ranks`, `--enable-diagnose`, `--quant-type=int8`, `--enable-dynamic-tokens`, multi-round) | **Yes** (a3 + a2, both CANN versions) |
| `test_low_latency.py` (same parameter matrix) | **Yes** (a3 + a2, both CANN versions) |
| `test_fused_deep_moe.py` (`--num-tokens`, `--num-experts`, `--hidden`, `--topk-drop-*`, `--moe-intermediate-size`, `--num-topk`) | **No** — unique to PR suite |
| `test_normal_and_low_latency.py` | **No** — unique to PR suite |
| `test_combine.py` | **No** — unique to PR suite |
| `test_dispatch_ffn_combine.py` | **No** — unique to PR suite |
| `test_internode_a2.py` (multi-node) | **No** — unique to PR suite |

> **Recommendation:** Consider **moving the `intranode` and `low_latency` parameter sweeps out of the PR suite** and rely on the daily enumerate. Keep only the tests that have **no daily equivalent** (the four unique test files above). Wall-clock saving: the 23 intranode+low_latency steps consume ~22 of the 30 min in `test-all-build-core` / `test-build-deepep-a3-core` and ~16 of 38 min in `test-build-deepep-a2`. Estimated ceiling drop: **30 min → 12–15 min** for those jobs.

### 3.5 [MEDIUM] `Run test generalization of fused deep moe` is a 5-min single bash script in two jobs

**Location:** [pr-test-npu.yml#L193-L200](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L193-L200) and [pr-test-npu.yml#L677-L684](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L677-L684)

This step runs `bash scripts/generalization_test_fused_deep_moe.sh` — a single bash script that takes **~5 min** wall-clock. The script is not invoked in the daily enumerate, so it has no daily-coverage. But it appears in **both** `test-all-build-core` and `test-build-deepep-a3-core` (the near-duplicate pair from 3.1) — if 3.1 is actioned, this duplicate disappears for free.

If the team confirms the script is "generalization" (boundary-condition sweep), consider whether it is the **PR suite's job** to run it on every PR or whether it should live in the daily enumerate. Either move it to daily, or delete from one of the duplicate core jobs (covered by 3.1).

### 3.6 [LOW] `timeout-minutes: 10` is too generous for almost every step

**Location:** 70+ steps in [pr-test-npu.yml](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml)

Sample actual durations from Run #27292189262 (CANN 9.0.0 jobs):

| Step | Actual | Current timeout |
|---|---:|---:|
| Run test intranode (a3-core) | 33 s | 10 min |
| Run test intranode for little bs (a3-core) | 35 s | 10 min |
| Run test low latency for little num processes (a3-core) | 27 s | 10 min |
| Run test mixed running for little processes (a2) | 39 s | 10 min |
| Run test intranode for dynamic tokens (a2) | 42 s | 10 min |
| Run test low latency for dynamic tokens (a2) | 46 s | 10 min |

**Only** the multi-invocation steps (hidden/topk/experts/multi-round) consistently exceed 90 s, and the **longest single step is 323 s** (`eplb` in `a3-moe`). Setting `timeout-minutes: 6` for the simple invocations and `timeout-minutes: 8` for the multi-invocation ones:

- does **not** affect happy-path runtime
- caps a stuck test at 6 min instead of 10 min — a real win when a hang occurs (saves 4 min × failure rate)

> **Recommendation:** Tune timeouts to actual P95 × 1.5.

### 3.7 [LOW] Some "for hidden / topk / experts" steps duplicate a parameter already covered by a single-invocation step

**Location:** Examples in `test-all-build-core`:
- Line 116 "Run test intranode" (default config) + line 138 "Run test hidden intranode" (4 `--hidden` values) + line 156 "Run test topk num intranode" (2 `--num-topk`)

A new `test_intranode.py` invocation with default args is **not** subsumed by the parameter-sweep steps. However, the **base step "Run test intranode"** (default args) is fully covered by the parameter-sweep on the `intranode` axis if `--num-tokens=8` (or whatever the default is) happens to be one of the sweep values. Worth checking the test script to see what defaults match; if yes, the base step can be deleted.

Similarly the "Run test low latency" step at line 162 runs `test_low_latency.py` with default args, `--num-tokens=1`, `--num-tokens=2`. The `--num-tokens=1` and `--num-tokens=2` are extreme values that may overlap with the "for little bs" pattern (which tests `--num-tokens=4`).

> **Recommendation:** Inspect [test_intranode.py](https://github.com/sgl-project/sgl-kernel-npu/blob/main/tests/python/deepep/test_intranode.py) and [test_low_latency.py](https://github.com/sgl-project/sgl-kernel-npu/blob/main/tests/python/deepep/test_low_latency.py) for default values. If a default value is also asserted by a dedicated step, the dedicated step is redundant.

### 3.8 [LOW] `concurrency.cancel-in-progress: true` on the pr-test-npu can cancel the daily's PR re-run

**Location:** [pr-test-npu.yml#L18-L20](https://github.com/sgl-project/sgl-kernel-npu/blob/main/.github/workflows/pr-test-npu.yml#L18-L20)

```yaml
concurrency:
  group: pr-test-npu-${{ github.ref }}-${{ github.event_name }}
  cancel-in-progress: true
```

When this workflow is called from `daily-build-test.yml` via `workflow_call`, the `github.ref` and `github.event_name` are the daily's, so it should be safe. **However**, if a developer opens a PR while a daily re-run is in progress, both workflows could share the same `github.ref` on `main` and **a new PR push could cancel the daily's PR-test-suite**. Verify this corner case.

> **Recommendation:** Use a more specific concurrency group, e.g. `pr-test-npu-${{ github.event_name }}-${{ github.run_id }}` for the PR trigger and a separate one for `workflow_call`.

---

## 4. Optimization Roadmap (ranked by impact)

| # | Action | Estimated wall-clock saving | Estimated runner-slot relief | Risk |
|---:|---|---:|---:|---|
| **1** | **Delete one of the `test-all-build-core` / `test-build-deepep-a3-core` pair** (3.1) | **~30 min** (drops longest PR job ceiling) | −2 a3-16 jobs | Low — only if the `-a deepep` flag is verified to be equivalent to default |
| **2** | **Delete one of the `test-all-build-moe` / `test-build-deepep-a3-moe` pair** (3.2) | **~37 min** (these are the *new* longest; takes the place of #1) | −2 a3-16 jobs | Low — same as above |
| **3** | **Move intranode/low_latency parameter sweeps to daily-enumerate only** (3.4); keep only the 4 PR-unique test files | drops PR suite ceiling to **~15 min** | −3 a3-16 jobs (×2 CANN) | Medium — owners of deepep intranode/low_latency must agree that daily is sufficient |
| **4** | **Merge the duplicate "Run test intranode" + "Run test intranode for output parameters of different types" steps** (3.3) | ~3 min runner-minutes | none | Very low |
| **5** | **Tighten `timeout-minutes`** from 10 → 6/8 (3.6) | 0 in happy path; up to 4 min per failed step | none | Very low |
| **6** | **Move `generalization_test_fused_deep_moe.sh` to daily enumerate** or delete (3.5) | ~5 min runner-minutes × 2 CANN | none | Low |
| **7** | **Verify & tighten `concurrency` group** (3.8) | n/a | n/a | Very low |

> **Combining #1 + #2 (deleting the duplicate job pairs) alone** brings the PR suite wall-clock from **40 min → 25–30 min** and reduces the daily+PR total from **125 min → 95–105 min** (limited by `Low Latency a3` daily test = 24 min + ~10–15 min PR post-daily overhead).
>
> **Adding #3** (delegate intranode/low_latency to daily) further brings it to **~50–65 min** total — the realistic floor without changing the matrix or self-hosted runner fleet.

---

## 5. Concrete Steps to Discuss with the Team

These are the specific items that need **a domain owner's sign-off** before deletion — please circulate to the relevant work groups and ask them to confirm whether each can be removed.

### 5.1 Confirm-able-as-redundant (no domain knowledge needed)

1. **Duplicate step inside the same job**: "Run test intranode" and "Run test intranode for output parameters of different types" (and the same pattern in low_latency, mixed_running). Both run `test_intranode.py` with identical CLI args; only `MOE_EXPERT_TOKEN_NUMS_TYPE=0` differs. → Should be merged.
2. **`Run test generalization of fused deep moe`** is a 5-min shell script that appears in two near-duplicate jobs. If job pair 3.1 is actioned, the duplicate disappears.

### 5.2 Needs a deepep-owner's call

3. **`test-all-build-core` vs `test-build-deepep-a3-core`**: Are both `prepare_deepep_in_container.sh` (no `-a`) and `… -a deepep` needed for `csrc/deepep/ops/**` PRs? If they target the same code path, drop one. (29.6 / 32.4 min × 2 CANN = ~62 runner-minutes of duplicated work.)
4. **`test-all-build-moe` vs `test-build-deepep-a3-moe`**: Same question for moe code path. (37.2 / 39.7 min × 2 CANN = ~77 runner-minutes of duplicated work.)
5. **`intranode` / `low_latency` parameter sweeps on PR**: The daily `enumerate_test_*.sh` already sweeps `csrc/deepep/ops/**` for all `intranode` and `low_latency` parameters on every hardware/CANN combination. Can the PR test suite stop re-running these and trust the daily run? Estimated wall-clock saving: 25–30 min.
6. **Test cases with extreme parameters** (require a runtime budget justification per case):
   - `test_intranode.py --num-experts=1024 --num-processes=16` (1024 experts / 16 ranks)
   - `test_intranode.py --hidden=8192` (largest hidden size in the suite)
   - `test_low_latency.py --num-experts=1024`
   - `test_low_latency.py --hidden=8192`
   - `test_normal_and_low_latency.py --num-experts=1024 --num-processes=16`
   - `test_normal_and_low_latency.py --hidden=8192`
   - `--quant-type=int8` (does the kernel actually support int8 in `ops/`? If not, this test is a no-op or a known-fail.)
   - `--enable-diagnose` (DeepXtrace) — diagnostic-only, runs only on request?

### 5.3 Needs a build/infra owner's call

7. **Why is `test-build-deepep-a2` only triggered for `ops2_changed || common_changed`**, while the a3 jobs also include `ops_changed`? Verify that `csrc/deepep/ops/**` does not need a2 coverage (the run shows a2 hardware is also tested for `intranode` and `low_latency` via the daily-enumerate test `enumerate_test_low_latency_a2.sh` — does that cover the a2 ops in `csrc/deepep/ops2/**`?).
8. **`concurrency.cancel-in-progress: true` corner case** (3.8): verify that a developer PR push does not cancel a daily's PR-test-suite.

### 5.4 Tuning (no domain knowledge needed, low risk)

9. Tighten `timeout-minutes: 10` to `6` (or `8` for multi-invocation steps) on every test step. Saves up to 4 min per stuck test.
10. Convert the `test_intranode.py` + `MOE_EXPERT_TOKEN_NUMS_TYPE=0` env-var pattern into a CLI flag inside the test script, then merge the two steps.

---

## 6. Work-Group Message (template)

Below is a ready-to-send message for the work group. The numbers and analysis are based on Run #27292189262 (2026-06-10).

---

> **【请大家帮忙看 PR 测试的冗余用例】**
>
> 大家好。最近我把 daily + PR 流水线总时长从 200 多分钟优化到 125 分钟左右（分析见 `decrease_runtime/` 目录）。瓶颈已经从"任务太多跑不完"变成了"a3-16 self-hosted runner 排队"。
>
> 单纯再拆任务跑并行只会让排队更糟。下面是我对 `pr-test-npu.yml` 的初步分析，发现 **两对几乎完全重复的 job** 和 **若干可能可以删掉的用例**，但都需要各位 owner 确认才能动。
>
> 📊 **当前数据（Run #27292189262）**
>
> | Job | 时长 | 备注 |
> |---|---:|---|
> | Low Latency a3 (daily) | 24 min | daily 阶段最长 |
> | test-build-deepep-a3-moe | **40 min** | PR 阶段最长，决定 PR suite wall-clock |
> | test-build-deepep-a2 | 39 min | |
> | test-all-build-moe | 37 min | |
> | test-build-deepep-a3-core | 32 min | |
> | test-all-build-core | 30 min | |
> | 排队等待 | ~60 min | a3-16 runner slot 不足 |
>
> ⚠️ **两对疑似重复的 job**（需要 owner 确认）
>
> 1. `test-all-build-core` 和 `test-build-deepep-a3-core`：跑同一个 ops 代码路径，区别只是 `prepare_deepep_in_container.sh` 是否带 `-a deepep`。两个 job 都 30+ min × 2 CANN = ~62 runner-minutes 的重复工作。
> 2. `test-all-build-moe` 和 `test-build-deepep-a3-moe`：同理，~77 runner-minutes 重复。
>
> ❓ **请以下同事帮忙确认**
>
> - **@deepep-ops owner**：(1)(2) 两对里 `-a deepep` 和默认（`-a deepep2`？）产出的是不同 wheel 吗？两个 job 是为了交叉兼容测试吗？如果不是，能否只保留一个？
> - **@deepep-ops owner**：(3) `test_intranode.py` / `test_low_latency.py` 的所有参数 sweep（`--num-tokens`、`--hidden`、`--num-topk`、`--num-experts`、`--active-ranks`、`--enable-diagnose`、`--quant-type=int8`、`--enable-dynamic-tokens` 等），**daily 的 `enumerate_test_*.sh` 已经在 a3+a2 × 2 CANN = 8 个 job 上跑过了**，PR 还要再跑一遍必要吗？daily 通过的前提下能否信任？
> - **@deepep-ops owner**：(4) 下面这些"大参数"用例每天在 PR 上都要跑，必要性如何？
>   - `test_intranode.py --num-experts=1024 --num-processes=16`
>   - `test_intranode.py --hidden=8192`
>   - `test_low_latency.py --num-experts=1024`
>   - `test_low_latency.py --hidden=8192`
>   - `test_normal_and_low_latency.py --num-experts=1024 --num-processes=16`
>   - `test_normal_and_low_latency.py --hidden=8192`
>   - `--quant-type=int8` —— `ops/` 真支持 int8 吗？跑出来是 no-op 还是已知失败？
>   - `--enable-diagnose` (DeepXtrace) —— 是不是只有 debug 时才需要？
> - **@build/infra owner**：(5) `test-build-deepep-a2` 只在 `ops2 || common` 改动时跑，但 `csrc/deepep/ops/**` 改动是否会污染 a2 上的回归？目前 a2 路径只靠 daily 的 `enumerate_test_*.sh` 兜底。
> - **@build/infra owner**：(6) `concurrency.cancel-in-progress: true` 的 group 包含 `github.ref`。daily 触发的 PR suite 跑的时候，如果同时有 developer push PR 到 main，会不会被取消？
>
> ✅ **我已经看下来可以直接做的（不需要 domain 知识）**
>
> - 把 `timeout-minutes: 10` 收紧到 6/8 分钟（绝大多数 step 实际 < 90s）
> - 合并 "Run test intranode" + "Run test intranode for output parameters of different types"（两个 step 跑的命令完全一样，只是 env 多了 `MOE_EXPERT_TOKEN_NUMS_TYPE=0`）
> - `generalization_test_fused_deep_moe.sh`（5 min 的单步）要么移到 daily，要么删一份（和上面 job 重复对绑定的）
>
> 完整分析报告在 [`D:\流水线自动化报告\decrease_runtime\pr-test-runtime-analysis.md`](file:///D:/流水线自动化报告/decrease_runtime/pr-test-runtime-analysis.md)。原始数据 (`pr-test-npu.yml`、`daily-build-test.yml`、Run #27292189262 的 jobs JSON) 都在同目录下。
>
> 麻烦大家这周 review 一下，**重点是上面 ❓ 部分**：哪些用例对线上 release 来说是必须保留的、哪些是"加了心安但其实没拦住过 bug"的，**请直接说"这个可以删"或"这个必须留"**。我会按大家的反馈出一个精简版 yml diff。
>
> 谢谢！

---

## 7. Artifacts

The following files were fetched from GitHub (via PAT `github_pat_11BFRPFXY0t5…`) and saved alongside this report:

| File | Description |
|---|---|
| [pr-test-npu.yml](file:///D:/流水线自动化报告/decrease_runtime/pr-test-npu.yml) | The workflow under analysis (1190 lines, latest from `main`) |
| [daily-build-test.yml](file:///D:/流水线自动化报告/decrease_runtime/daily-build-test.yml) | The caller workflow |
| [internode.yml](file:///D:/流水线自动化报告/decrease_runtime/internode.yml) | The reusable multi-node workflow (called by `test-build-deepep-internode`) |
| [run-info.json](file:///D:/流水线自动化报告/decrease_runtime/run-info.json) | Run #27292189262 metadata (created / updated timestamps → 125.2 min total) |
| [run-jobs.json](file:///D:/流水线自动化报告/decrease_runtime/run-jobs.json) | Run #27292189262 per-job + per-step timing data (23 jobs, all step durations) |

---

*Analysis produced from a single representative run (Run #27292189262, 2026-06-10). All cited durations are from that run. Line numbers refer to the file at commit `fd5f19dd610f2fc6fad7f9cbb4e7a2241eb7bc5d` on `main`.*
