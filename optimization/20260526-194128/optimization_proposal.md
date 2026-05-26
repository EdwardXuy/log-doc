# CI/CD Pipeline Optimization Proposal

Generated: 2026-05-26 20:27:00
Scope: Reduce total pipeline execution time without modifying test scripts

## 1. Current PR Workflow Time Breakdown

| Job Name | Count | Avg (min) | Max (min) |
|----------|-------|-----------|-----------|
| multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2 | 90 | 25.7 | 181.6 |
| test-build-deepep-a2 | 90 | 28.8 | 152.1 |
| test-all-build | 90 | 45.8 | 101.6 |
| test-build-deepep-a3 | 90 | 44.4 | 71.1 |

## 2. Bottleneck Analysis

The PR workflow has the following time bottlenecks (sorted by avg duration):

- **test-all-build**: avg 45.8 min, max 101.6 min
- **test-build-deepep-a3**: avg 44.4 min, max 71.1 min
- **test-build-deepep-a2**: avg 28.8 min, max 152.1 min
- **multi-node (test_internode_a2, 2, tests/python/deepep/test_internode_a2.py) / test_internode_a2**: avg 25.7 min, max 181.6 min

## 3. Optimization Proposals

### Proposal 1: Split test-all-build into Parallel Sub-Jobs

Current: test-all-build runs all DeepEP tests sequentially (45+ steps, ~63 min)
Proposal: Split into 3 parallel jobs by test type:

| New Job | Tests Included | Est. Duration |
|---------|---------------|--------------|
| test-intranode | test_intranode (13 variants) | ~15 min |
| test-low-latency-moe | test_low_latency (8) + test_fused_deep_moe (8) + test_mixed_running (7) | ~25 min |
| test-combine-misc | test_combine (1) + test_generalization_fused_deep_moe (1) | ~10 min |

Expected improvement: Wall-clock time from ~63 min to ~25 min (60% reduction)
Implementation: Modify pr-test-npu.yml, add matrix strategy or separate jobs

### Proposal 2: Share Build Artifact Across Jobs

Current: test-all-build, test-build-deepep-a3, test-build-deepep-a2 each build DeepEP independently (~10-15 min build time each)
Proposal: Create a dedicated build job, upload wheel as GitHub Actions artifact, other jobs download artifact

Expected improvement: Save 20-30 min of redundant build time per run
Implementation: Add actions/upload-artifact and actions/download-artifact steps

### Proposal 3: Conditional Internode Testing

Current: test_internode_a2 runs on every PR, but fails ~90% of the time due to K8s resource issues, blocking the entire workflow for up to 3 hours
Proposal:
- Option A: Set continue-on-error: true for internode job so it does not block the workflow
- Option B: Only run internode on schedule (daily) or manual trigger, not on every PR
- Option C: Reduce timeout from 10800s (3h) to 1800s (30min)

Expected improvement: PR workflow wall-clock time from 3+ hours to <30 min when K8s is unavailable

### Proposal 4: Optimize enumerate_test Shell Scripts

Current: scripts/enumerate_test_intranode.sh and enumerate_test_low_latency.sh run tests sequentially for each parameter combination
Proposal: Run parameter combinations in parallel within the job using background processes or GNU parallel

Expected improvement: 30-50% reduction in daily-build-test job duration

### Proposal 5: Add sgl_kernel_npu Operator Tests to CI

Current: 36 operator tests in tests/python/sgl_kernel_npu/ are not in any CI workflow
Proposal: Add a new job in daily-build-test.yml or pr-test-npu.yml:

```yaml
  test-kernel-operators:
    runs-on: linux-aarch64-a3-16
    steps:
      - uses: actions/checkout@v4
      - run: scripts/npu_ci_install_dependency.sh
      - run: ./build.sh
      - run: pip install -e .
      - run: pytest tests/python/sgl_kernel_npu/ -v --timeout=300
```

Note: These tests need NPU hardware, so they must run on self-hosted runners with Ascend NPU

## 4. Summary of Expected Improvements

| Proposal | Target | Expected Time Saving | Priority |
|----------|--------|---------------------|----------|
| Split test-all-build | PR workflow | ~38 min (63 to 25 min) | P0 |
| Share build artifact | PR workflow | ~20 to 30 min | P0 |
| Conditional internode | PR workflow | ~2.5 hours (when K8s down) | P0 |
| Parallel enumerate | Daily workflow | 30 to 50 pct job time | P1 |
| Add operator tests | Coverage | N/A (new tests) | P1 |
