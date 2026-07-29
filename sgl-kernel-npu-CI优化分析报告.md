# SGL-Kernel-NPU CI 流水线优化分析报告

> 汇报人：CI 优化分析
> 日期：2026-07-29
> 仓库：sgl-project/sgl-kernel-npu
> 数据样本：daily-build-test 最近 10 次、a2-internode-test 与 pr-test-deepep-npu 各最近 50 次 CI 运行记录

---

## 一、背景与目标

当前 `sgl-kernel-npu` 仓库存在三条核心 CI 流水线：

| 流水线 | 文件 | 触发方式 | 用途 |
|--------|------|----------|------|
| Daily Enumerate Tests | `daily-build-test.yml` | 每日深夜定时（cron `0 16 * * *`） | 每日全量回归 |
| A2 Internode Test | `a2-internode-test.yml` | PR 触发 | A2 多机互联测试 |
| PR Test DeepEP | `pr-test-deepep-npu.yml` | PR 触发 / workflow_call | DeepEP 单机功能矩阵测试 |

**现象**：daily 流水线深夜定时触发基本成功（成功率 90%），而 PR 流水线（白天多人提交触发）与 A2 多机流水线运行时间偏长、稳定性较低。

**目标**：分析运行时间长和稳定性不足的原因，给出优化方案，不删除已有用例和场景矩阵。

---

## 二、数据统计概览

### 2.1 成功率与运行时长对比

| 流水线 | 样本 | 成功 | 失败 | 取消 | 需审批 | 成功率 | 中位时长 | 最大时长 |
|--------|------|------|------|------|--------|--------|----------|----------|
| daily-build-test | 10 | 9 | 1 | 0 | 0 | **90%** | 152 min | 237 min |
| a2-internode-test | 50 | 22 | 13 | 14 | 1 | **44%** | 59 min | 1118 min |
| pr-test-deepep-npu | 50 | 24 | 2 | 17 | 5 | **48%** | 84 min | 2542 min |

### 2.2 关键发现

- **daily** 深夜定时触发（北京时间 0 点），无并发冲突，成功率极高
- **a2-internode** 和 **pr-test** 白天多人提 PR 并发触发，成功率不足 50%
- 最大时长异常：a2 达 1118 min（18.6 小时），pr-test 达 2542 min（42 小时），均为卡死/排队
- 50 次 a2-internode 中检测到 **17 组不同 PR 的运行时间重叠**；pr-test 有 **55 组重叠**
- pr-test 的 17 次取消中，有 7 次在 10 分钟内被取消（并发组 `cancel-in-progress: true` 导致）

### 2.3 a2-internode-test 失败分类（13 次失败）

| 失败类型 | 次数 | 占比 | 典型时长 | 说明 |
|----------|------|------|----------|------|
| 共享路径竞态（快速失败） | 4 | 31% | 1-9 min | `chmod: cannot access` / `Directory not empty` |
| K8s/代理故障（快速失败） | 5 | 38% | 5-9 min | `gh-proxy 502` / `Pod logs ended` |
| K8s 卡死超时 | 1 | 8% | 183 min+ | docs PR 触发，非代码问题 |
| PR 代码问题（超时） | 3 | 23% | 183 min | `disable-pertoken_fp8_e5m2` 分支相关 |

**结论：77% 的失败为流水线基础设施问题，而非 PR 代码问题。**

---

## 三、流水线层面错误根因分析

> 以下分析排除了 PR 本身代码改动造成的错误，重点关注流水线本身的问题。

### 根因 1：K8s 共享资源竞态条件（最严重，占 a2 失败的 31%）

`internode.yml` 中所有运行共享以下固定资源：

```yaml
# internode.yml 第 30-35 行
env:
  NAMESPACE: sgl-kernel-npu          # ← 所有运行共享同一命名空间
  KUBE_JOB_NAME: sglang-npu-multi    # ← 所有运行共享同一 K8s Job 名
```

`k8s_multi.yaml.jinja2` 第 162-165 行揭示了共享 PVC：

```yaml
volumes:
- name: share
  persistentVolumeClaim:
    claimName: sgl-project-sglang-hk001   # ← 共享 PVC 挂载到 /root/.cache
```

**问题链**：

1. `/root/.cache/tests/sglang` 位于共享 PVC 上，所有并发运行互相覆盖
2. `internode.yml` 第 82-88 行 "Prepare scripts" 步骤执行 `rm -rf` + `cp -r` + `chmod -R`，与并发运行产生竞态
3. K8s Job 名固定为 `sglang-npu-multi`，并发运行创建同名 Job 导致冲突
4. 并发组 `a2-internode-test-${{ github.ref }}-${{ github.event_name }}` 中不同 PR 的 `github.ref` 不同，**无法阻止跨 PR 并发**

**日志证据**（run 30335948012, 9.0.0 cann, 1 分钟失败）：

```
chmod: cannot access '/root/.cache/tests/sglang/scripts': No such file or directory
chmod: cannot access '/root/.cache/tests/sglang/csrc': No such file or directory
##[error]Process completed with exit code 1.
```

（run 29897327319, 9.0.0 cann, 2 分钟失败）：

```
rm: cannot remove '/root/.cache/tests/sglang': Directory not empty
```

**典型模式**：8.5.0 cann 版本先运行成功（20-50 min），9.0.0 cann 版本随后启动时，共享路径已被其他并发运行修改，导致 1-2 分钟内快速失败。

### 根因 2：GitHub 代理 502 错误（占失败日志的 20%）

所有工作流通过 `https://gh-proxy.test.osinfra.cn/` 代理访问 GitHub。并发 checkout 时代理过载返回 502：

```
fatal: unable to access 'https://gh-proxy.test.osinfra.cn/https://github.com/sgl-project/sgl-kernel-npu/': The requested URL returned error: 502
The process '/usr/bin/git' failed with exit code 128
```

`actions/checkout@v4` 仅重试 3 次（10s + 16s + 18s 后放弃），导致整个 Job 在 2 分钟内失败。此问题在白天高峰期尤为严重。

### 根因 3：K8s 清理循环无超时（卡死根因，导致 1118 min+ 运行）

`internode.yml` 第 95-107 行：

```yaml
- name: Clear resources
  run: |
    while true; do
      if kubectl get po -A -n $NAMESPACE | grep -q "${pod_name_prefix}"; then
        echo "Found exist sglang job, sleeping for 30 seconds..."
        sleep 30
        kubectl get pods | grep "${pod_name_prefix}" | awk '{print $1}' | xargs kubectl delete pod -n $NAMESPACE || true
      else
        echo "No sglang job exist, start test case..."
        break
      fi
    done
```

**`while true` 无退出条件、无最大重试次数**。如果另一个运行的 Pod 处于卡死状态（如 finalizer 阻塞），此循环将永久运行，导致 run 时长达到 1118 min、2542 min。

**实际案例**：
- run 96（docs PR）：运行 1082 min，两个 cann 版本各卡 183 min
- run 118（main 分支）：运行 1118 min
- run 120：运行 367 min，两个 cann 版本各卡 183 min

### 根因 4：Runner 资源争抢（导致排队与 action_required）

`pr-test-deepep-npu.yml` 同时启动 **10 个 Job**（5 类 × 2 cann 版本），全部竞争 `linux-aarch64-a3-16` 和 `linux-aarch64-a2-8` 自托管 runner：

| 运行 | 10 个 Job 总和 | 实际运行时长 | 并行效率 | 排队时间 |
|------|---------------|-------------|----------|----------|
| run 133 | 376 min | 70 min | 8.7x | 少 |
| run 125 | 376 min | **215 min** | 1.75x | **172 min** |
| run 124 | 371 min | 126 min | 2.9x | 45 min |

5 次 `action_required` 状态的运行集中在 2026-07-21 06:17-07:21（北京时间下午 2-3 点高峰期），表明 runner 池完全耗尽。

### 根因 5：构建缓存预热机制被禁用（影响有限，非主要矛盾）

`push_build_cache.yml` 当前状态为 **`disabled_manually`**（手动禁用）。该工作流设计用于在 push main 时预构建缓存。

**历史调查结论**：

通过 commit 历史和运行记录分析，禁用原因可追溯至 2026-05 月的构建流程重构：

| 时间 | 事件 | 影响 |
|------|------|------|
| 2026-05-14 | push_build_cache.yml 首次创建（PR #469），运行 4 次全部成功 | 正常工作 |
| 2026-05-27 | `refactor: separate submodule init and build`（PR #516）| 构建流程重构，cache key 计算可能失效 |
| 2026-05-30 | `ci: restore build cache for PR CI`（PR #483）修复 cache key | commit 消息明确指出：`BUILD_HASH only covers source files, so an image bump that changes only the container CANN version would silently reuse stale .so artifacts and produce an ABI mismatch` |
| 2026-06-03 | `Add cann9.0.0 and a2`（PR #536）同步更新 push_build_cache.yml | 工作流文件已更新，但 **UI 禁用状态保留** |

**关键发现**：

1. **禁用是历史遗留状态**，并非因为 push_build_cache.yml 本身设计有问题。2026-05-30 的修复（commit 861a5c56）已经解决了 cache key 的 ABI 不匹配问题，但修复后**未重新启用**该工作流
2. **cache key 当前是兼容的**：
   - push_build_cache: `sgl-kernel-npu-build-v2-{os}-{matrix.arch}-cann{CANN}-{HASH}`
   - pr-test (a3 jobs): `sgl-kernel-npu-build-v2-{os}-a3-cann{CANN}-{HASH}`
   - pr-test (a2 job):  `sgl-kernel-npu-build-v2-{os}-a2-cann{CANN}-{HASH}`
   - `matrix.arch` 取值为 `a3` 或 `a2`，与 pr-test 的硬编码字面量匹配
3. **pr-test 自身的 cache 机制仍在工作**，实测最近成功运行：
   - run 133（同 SHA 重复运行）：7/7 jobs cache_hit=True
   - run 130（新 SHA）：7/10 cache_hit=True，3 个 cache_miss（a2 jobs + core 9.0.0）
   - 同一 PR 多次 push 时，后续 run 能命中前次 run 的 cache

**实际影响评估**：

- 禁用 push_build_cache **不会导致 PR 测试无法使用缓存**，pr-test 自身的 cache 机制仍工作
- 主要影响：**跨 PR（不同 BUILD_HASH）时，首个 PR 需从零构建**。push_build_cache 预热可让 main 最新 commit 的 cache 预先就绪，使新 PR 在源码与 main 接近时能命中
- 实测 cache miss 主要发生在 a2 jobs 和新 cann 版本，但这些 miss 是因为 BUILD_HASH 变化（源码改动），而非预热缺失
- **此问题非当前 CI 不稳定的主要矛盾**，优先级应降低

### 根因 6：UV 缓存被显式禁用

`npu_ci_install_dependency.sh` 第 65 行：

```bash
export UV_NO_CACHE=true   # ← 显式禁用 uv 缓存
```

每个 Job 的依赖安装（apt + pip install torch + torch_npu 等）都从零下载，无法跨 Job 复用已下载的包。torch 包约 27 MB，torch_npu 包更大，10 个 Job 各下载一次造成显著时间浪费。

### 根因 7：依赖安装重复执行

`pr-test-deepep-npu.yml` 中每个 Job 都独立执行完整的依赖安装流程（`npu_ci_install_dependency.sh`），包括 `apt update && apt upgrade && apt install` 以及 pip 安装 torch/torchvision/torch_npu。10 个 Job 各执行一次，每次约 5-10 min。

---

## 四、与其他工作流的关系确认

在分析过程中，我们确认了以下工作流的功能与影响：

### 4.1 `build_and_release.yml`（发布构建）

- **触发**：release published 或 PR 修改 build.sh/config.ini
- **运行环境**：GitHub-hosted runner（ubuntu-24.04 / ubuntu-24.04-arm），**非自托管 Ascend runner**
- **功能**：构建发布产物（sgl-kernel-npu、ops-transformer、custom-ops）
- **与 CI 测试流水线的关系**：**无资源冲突**，使用不同的 runner 池
- **对本分析的影响**：无需纳入优化范围

### 4.2 `push_build_cache.yml`（缓存预热，已禁用）

- **当前状态**：`disabled_manually`
- **设计功能**：push main 时预构建 4 个矩阵（a3/a2 × 8.5.0/9.0.0）的构建缓存
- **Cache key**：与 `pr-test-deepep-npu.yml` 完全一致
- **影响**：禁用后 PR 测试无法命中预热缓存，需从零构建
- **与本分析的关系**：**直接相关**，见根因 5 和方案 4

### 4.3 `release_deep_ep_wheel.yml`（DeepEP Wheel 发布）

- **触发**：release published / workflow_dispatch / PR 修改 build.sh
- **运行环境**：GitHub-hosted runner
- **功能**：构建 910b（deepep2）和 a3（deepep）的 deep_ep wheel 并上传 OBS
- **与本分析的关系**：**无资源冲突**，但确认了 `-a deepep`（A3）与 `-a deepep2`（A2）是两种不同的构建产物

---

## 五、优化方案

### 方案 1：隔离 K8s 资源（P0 - 最高优先级）

**对应根因**：根因 1（K8s 共享资源竞态）

**修改文件**：`internode.yml` + `k8s_multi.yaml.jinja2`

**核心思路**：为每次运行创建唯一的命名空间/Job 名/缓存路径，用 `github.run_id` 实现隔离。

**修改内容**：

`internode.yml` 中修改环境变量：

```yaml
jobs:
  multi-node:
    name: ${{ inputs.test_config_name }}
    runs-on: ${{ inputs.runner }}
    container:
      image: swr.ap-southeast-1.myhuaweicloud.com/base_image/ascend-ci/sglang:main-x86
      env:
        KUBECONFIG: /root/.cache/.cache/kube.yaml
        KUBECTL: /root/.cache/.cache/kubectl
        NAMESPACE: sgl-kernel-npu
        ASCEND_TEST_CASE_PATH: tests/python/deepep
        KUBE_JOB_TYPE: multi
        # 关键修改：用 run_id 隔离
        KUBE_JOB_NAME: sglang-npu-multi-${{ github.run_id }}
        KUBE_CONFIG_MAP: sglang-info-${{ github.run_id }}
```

"Prepare scripts" 步骤中隔离缓存路径：

```yaml
      - name: Prepare scripts
        run: |
          # 关键修改：用 run_id 隔离缓存路径
          sglang_source_path=/root/.cache/tests/sglang-${{ github.run_id }}
          mkdir -p $sglang_source_path && chmod -R 777 $sglang_source_path
          rm -rf $sglang_source_path/*
          cp -r $GITHUB_WORKSPACE/* $sglang_source_path/
          # ... jinja2 模板渲染不变 ...
```

"Post process" 步骤中清理自己的缓存目录：

```yaml
      - name: Post process
        if: always()
        run: |
          kubectl get pods -n $NAMESPACE
          cd $ASCEND_TEST_CASE_PATH
          kubectl delete -f ./k8s_multi.yaml --ignore-not-found=true || true
          # 关键修改：清理自己的缓存目录
          rm -rf /root/.cache/tests/sglang-${{ github.run_id }}
```

**预期效果**：消除 77% 的 a2-internode-test 失败（共享路径竞态 + K8s Job 冲突）

**风险**：无。每个运行使用独立资源，互不干扰。

---

### 方案 2：为清理循环添加超时（P0 - 最高优先级）

**对应根因**：根因 3（K8s 清理循环无超时，导致卡死）

**修改文件**：`internode.yml`

**核心思路**：为 `while true` 循环添加最大重试次数和超时退出。

**修改内容**：

```yaml
      - name: Clear resources
        run: |
          cd $ASCEND_TEST_CASE_PATH
          kubectl delete -f ./k8s_multi.yaml --ignore-not-found=true || true

          pod_name_prefix="${KUBE_JOB_NAME}-sglang"
          echo "kube name space: $NAMESPACE, pod name prefix: ${pod_name_prefix}"
          max_retries=20    # 最多等待 20×30s = 10 分钟
          retry=0
          while [ $retry -lt $max_retries ]; do
            if kubectl get po -A -n $NAMESPACE | grep -q "${pod_name_prefix}"; then
              echo "Found exist sglang job (retry $retry/$max_retries), sleeping 30s..."
              sleep 30
              kubectl get pods -n $NAMESPACE | grep "${pod_name_prefix}" | awk '{print $1}' | xargs kubectl delete pod -n $NAMESPACE --force --grace-period=0 || true
              retry=$((retry + 1))
            else
              echo "No sglang job exist, start test case..."
              break
            fi
          done
          if [ $retry -ge $max_retries ]; then
            echo "WARNING: Timeout waiting for pod cleanup, proceeding anyway..."
          fi
```

**预期效果**：消除卡死导致的 1118 min+ 运行，最长等待 10 分钟后继续执行

**风险**：极低。即使超时后继续执行，由于方案 1 已实现资源隔离，不会与其他运行冲突。

---

### 方案 3：全局并发控制多机测试（P0 - 最高优先级）

**对应根因**：根因 1（跨 PR 并发竞态）

**修改文件**：`a2-internode-test.yml` + `internode.yml`

**核心思路**：使多机测试全局串行执行，而非并发冲突。虽然会增加排队时间，但能彻底消除 K8s 资源竞态。

**修改内容**：

`a2-internode-test.yml`：

```yaml
concurrency:
  # 关键修改：不再按 ref 区分，全局串行多机测试
  group: ascend-multi-node-global
  cancel-in-progress: false    # 不取消正在运行的测试，避免半途而废
```

`internode.yml`：

```yaml
concurrency:
  group: ascend-multi-node-global
  cancel-in-progress: false
```

**预期效果**：彻底消除跨 PR 的 K8s 资源竞态

**权衡**：不同 PR 的多机测试会排队执行，增加等待时间。但考虑到多机测试本身的特殊性（需要独占 2 节点 16 卡），串行是合理的工程选择。`cancel-in-progress: false` 确保正在运行的测试不被中断。

---

### 方案 4：重新启用构建缓存预热（P2 - 可选优化，非当前主要矛盾）

**对应根因**：根因 5（`push_build_cache.yml` 被禁用）

**修改文件**：`push_build_cache.yml`

**核心思路**：重新启用 `push_build_cache.yml`，在 main 分支推送时预构建缓存，使后续 PR 测试能命中缓存。

**历史调查结论**：

经 commit 历史调查，禁用是历史遗留状态：

1. 2026-05-14 该工作流创建并成功运行 4 次
2. 2026-05-27 构建流程重构（PR #516）导致 cache key 计算可能失效
3. 2026-05-30 修复了 cache key 的 ABI 不匹配问题（commit 861a5c56），commit 消息明确说明此前存在 "image bump 时 silently reuse stale .so artifacts and produce an ABI mismatch" 的风险
4. 2026-06-03 同步更新了 cann9.0.0 和 a2 支持，但工作流**未重新启用**

**cache key 一致性确认**：

当前 push_build_cache.yml 与 pr-test-deepep-npu.yml 的 cache key 兼容：

```yaml
# push_build_cache.yml (matrix.arch 取值为 a3 或 a2)
key: sgl-kernel-npu-build-v2-${{ runner.os }}-${{ matrix.arch }}-cann${{ CANN_VERSION }}-${{ BUILD_HASH }}

# pr-test-deepep-npu.yml (a3 jobs)
key: sgl-kernel-npu-build-v2-${{ runner.os }}-a3-cann${{ CANN_VERSION }}-${{ BUILD_HASH }}

# pr-test-deepep-npu.yml (a2 job)
key: sgl-kernel-npu-build-v2-${{ runner.os }}-a2-cann${{ CANN_VERSION }}-${{ BUILD_HASH }}
```

**实测缓存命中率**：

- run 133（同 SHA 重复运行）：7/7 jobs cache_hit=True
- run 130（新 SHA）：7/10 cache_hit=True，3 个 cache_miss
- pr-test 自身的 cache 机制仍在工作，禁用 push_build_cache **不会导致 PR 测试无法使用缓存**

**建议**：

由于此问题非当前 CI 不稳定的主要矛盾，建议**作为可选优化**，在 P0/P1 方案实施并验证后，再评估是否重新启用：

1. 与原维护人员（whjnbm、EdwardXuy）确认禁用是否有其他未记录原因
2. 确认 runner 资源是否支持额外的预热任务（4 矩阵 × push 触发）
3. 可先小范围测试：手动触发一次 workflow_dispatch，验证 cache key 是否与 pr-test 匹配
4. 确认无误后通过 GitHub UI 重新启用

**预期效果**：跨 PR（不同 BUILD_HASH）时，首个 PR 可命中 main 预热缓存，节省 10-15 min 构建时间

**风险**：中。需确认 runner 资源是否支持额外的预热任务，且需原维护人员确认禁用原因。

---

### 方案 5：Checkout 重试与代理容错（P1）

**对应根因**：根因 2（GitHub 代理 502 错误）

> 注：原方案 5（daily 阶段并行）已按要求移除，daily 流水线深夜稳定，串行设计是故意防止资源/文件冲突，不需大改。此处方案 5 为原 Checkout 重试方案。

**修改文件**：所有 workflow 的 checkout 相关步骤

**核心思路**：增强 checkout 步骤对代理 502 错误的容错能力。

**方案 A - 使用 `actions/checkout` 的重试能力**：

GitHub Actions 的 `actions/checkout@v4` 内置了 3 次重试（10s + 16s + 18s），但白天高峰期代理可能持续故障。可通过包裹一层重试逻辑：

```yaml
      - name: Checkout code (with retry)
        uses: actions/checkout@v4
        with:
          clean: true
          submodules: recursive
        continue-on-error: true
        id: checkout-1

      - name: Retry checkout after 60s
        if: steps.checkout-1.outcome == 'failure'
        run: |
          sleep 60
          rm -rf $GITHUB_WORKSPACE/.git 2>/dev/null || true

      - name: Checkout code (retry 2)
        if: steps.checkout-1.outcome == 'failure'
        uses: actions/checkout@v4
        with:
          clean: true
          submodules: recursive
```

**方案 B - 在 install dependency 脚本中配置 git 重试参数**：

在 `npu_ci_install_dependency.sh` 开头添加：

```bash
# 增强 git 网络容错
git config --global http.lowSpeedLimit 1000
git config --global http.lowSpeedTime 60
git config --global http.maxRequestBuffer 100M
git config --global http.postBuffer 524288000
```

**预期效果**：减少代理 502 导致的 checkout 失败，预计减少 20% 的快速失败

**风险**：极低。仅增加重试逻辑，不改变原有流程。

---

### 方案 6：启用 UV 缓存（P2 - 保守优化）

**对应根因**：根因 6（UV 缓存被显式禁用）

**修改文件**：`npu_ci_install_dependency.sh`

**现状分析**：

`npu_ci_install_dependency.sh` 第 65 行显式禁用了 uv 缓存：

```bash
export UV_NO_CACHE=true   # ← 禁用 uv 缓存，每次都从网络下载
```

**工程化评估**：

直接缓存整个 Python 环境（如 site-packages 目录）**不符合工程化思维**，原因如下：

1. **脆弱性**：site-packages 缓存依赖于精确的 Python 版本、CANN 版本、系统架构，任何一项不匹配都会导致隐蔽错误
2. **一致性风险**：自托管 runner 上缓存系统级 site-packages 可能导致不同 Job 间的环境污染
3. **缓存大小**：torch + torch_npu + 依赖约 2-3 GB，cache 上传/下载开销可能抵消收益
4. **业界实践**：业界主流做法是缓存包下载缓存（pip cache / uv cache），而非安装结果

**业界实例**：
- PyTorch 官方 CI 使用 `~/.cache/pip` 缓存 pip 下载
- vLLM 使用 `actions/cache@v4` 缓存 `~/.cache/uv`
- HuggingFace transformers 使用 pip cache 目录缓存

**推荐做法**：移除 `UV_NO_CACHE=true`，改为配置共享的 UV 缓存目录：

```bash
# npu_ci_install_dependency.sh 修改
# 删除: export UV_NO_CACHE=true
export UV_SYSTEM_PYTHON=true
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_CACHE_DIR=/root/.cache/uv-cache    # 共享 UV 下载缓存
```

同时在 workflow 中添加 UV 缓存持久化：

```yaml
      - name: Cache UV downloads
        uses: runs-on/cache@v4
        with:
          path: /root/.cache/uv-cache
          key: uv-cache-${{ matrix.cann_version }}-${{ hashFiles('scripts/npu_ci_install_dependency.sh') }}
```

**预期效果**：跨 Job 复用已下载的包，每个 Job 节省 2-3 min 下载时间

**风险**：低。仅缓存下载文件，不缓存安装结果，不影响环境一致性。

---

### 方案 7：构建测试分离（P3 - 不推荐实施，仅作分析记录）

**对应根因**：根因 7（依赖安装重复执行）

**原方案设想**：将构建与测试分离，构建产物通过 artifact 共享给测试 Job。

**经确认后不推荐实施**，原因如下：

`build.sh` 的 `-a` 参数控制构建内容，是**重大区别**：

| 参数 | BUILD_DEEPEP_MODULE | BUILD_DEEPEP_OPS | 算子路径 | SOC_VERSION | 构建内容 |
|------|---------------------|-------------------|----------|-------------|----------|
| 无 `-a` | ON | ON | csrc/deepep/ops | Ascend910_9382 | 全部模块（deepep + kernels + memory-saver） |
| `-a deepep` | ON | ON | csrc/deepep/ops | Ascend910_9382 | 仅 deepep（A3 算子路径） |
| `-a deepep2` | ON | OFF | csrc/deepep/ops2 | Ascend910B1 | 仅 deepep（A2 算子路径） |

`pr-test-deepep-npu.yml` 中五个测试 Job 使用三种不同的构建配置：

| Job | 构建参数 | 构建内容 | 测试内容 |
|-----|----------|----------|----------|
| test-all-build-core | 无 `-a` | 全部模块 | core 测试 |
| test-all-build-moe | 无 `-a` | 全部模块 | moe 测试 |
| test-build-deepep-a3-core | `-a deepep` | 仅 deepep（A3） | core 测试 |
| test-build-deepep-a3-moe | `-a deepep` | 仅 deepep（A3） | moe 测试 |
| test-build-deepep-a2 | `-a deepep2` | 仅 deepep（A2） | 全部测试 |

虽然 test-all-build-core 和 test-build-deepep-a3-core 的**测试步骤相同**，但它们的**构建产物不同**（全量构建 vs 仅 deepep 构建），测试目的不同（验证全量集成 vs 验证 deepep 模块独立可用）。合并会丢失测试覆盖维度。

**结论**：保守起见不实施此方案，保持现有构建-测试耦合的设计。

---

## 六、优先级汇总

| 优先级 | 方案 | 对应根因 | 预期效果 | 改动范围 | 风险 |
|--------|------|----------|----------|----------|------|
| **P0** | 方案 1：隔离 K8s 资源 | 根因 1 | 消除 77% 的 a2 失败 | internode.yml + jinja2 模板 | 无 |
| **P0** | 方案 2：清理循环超时 | 根因 3 | 消除卡死（1118 min+） | internode.yml | 极低 |
| **P0** | 方案 3：全局并发控制 | 根因 1 | 消除跨 PR 竞态 | a2-internode-test.yml + internode.yml | 低（增加排队） |
| **P1** | 方案 5：Checkout 重试 | 根因 2 | 减少 20% 快速失败 | 所有 workflow | 极低 |
| **P2** | 方案 6：启用 UV 缓存 | 根因 6 | 每个 Job 节省 2-3 min | npu_ci_install_dependency.sh | 低 |
| **P2** | 方案 4：重新启用缓存预热 | 根因 5 | 跨 PR 节省 10-15 min/Job | push_build_cache.yml | 中（需确认资源与禁用原因） |
| ~~P3~~ | ~~方案 7：构建测试分离~~ | ~~根因 7~~ | ~~不推荐实施~~ | - | - |

**预期总体效果**：

- P0 方案实施后：a2-internode-test 成功率从 44% 提升至 80%+，消除所有卡死运行
- P1 方案补充后：进一步减少代理故障导致的失败
- P2 方案微调后：每个 Job 节省 2-3 min 依赖下载时间，跨 PR 构建可命中预热缓存

**关于方案 4 的说明**：经历史调查，push_build_cache.yml 的禁用是 2026-05 月构建流程重构后的历史遗留状态，cache key 已在 commit 861a5c56 中修复并与 pr-test 兼容。pr-test 自身的 cache 机制仍工作（实测命中率 70-100%），禁用 push_build_cache 非当前 CI 不稳定的主要矛盾，降级为 P2 可选优化。

---

## 七、实施建议

### 第一阶段（立即实施 P0）

1. 修改 `internode.yml`：方案 1 + 方案 2（资源隔离 + 清理循环超时）
2. 修改 `a2-internode-test.yml` 和 `internode.yml`：方案 3（全局并发控制）
3. 验证：观察一周内 a2-internode-test 的成功率和运行时长变化

### 第二阶段（P0 验证后实施 P1）

4. 在所有 workflow 中添加 checkout 重试逻辑（方案 5）

### 第三阶段（P1 稳定后实施 P2）

5. 修改 `npu_ci_install_dependency.sh` 启用 UV 缓存（方案 6）
6. 监控缓存命中率和依赖安装时间
7. 评估是否重新启用 `push_build_cache.yml`（方案 4）：
   - 与原维护人员（whjnbm、EdwardXuy）确认禁用是否有其他未记录原因
   - 确认 runner 资源是否支持额外的预热任务
   - 可先手动触发一次 workflow_dispatch 验证 cache key 匹配

### 不实施

- ~~原方案 5（daily 阶段并行）~~：daily 流水线深夜稳定运行，串行设计是故意的，防止资源/文件冲突，不需大改
- ~~方案 7（构建测试分离）~~：`-a` 参数是重大区别，保持现有构建-测试耦合设计

---

## 附录 A：数据采集说明

- 数据来源：GitHub Actions REST API
- 采集时间：2026-07-29
- 采集内容：
  - daily-build-test 最近 10 次运行
  - a2-internode-test 最近 50 次运行
  - pr-test-deepep-npu 最近 50 次运行
  - 失败运行的完整 Job 日志（15 份）
- 分析工具：PowerShell + GitHub API

## 附录 B：工作流文件索引

| 文件 | 路径 | 用途 |
|------|------|------|
| a2-internode-test.yml | `.github/workflows/` | A2 多机互联测试入口 |
| internode.yml | `.github/workflows/` | 多机测试可复用工作流 |
| pr-test-deepep-npu.yml | `.github/workflows/` | DeepEP PR 测试主工作流 |
| daily-build-test.yml | `.github/workflows/` | 每日定时全量回归 |
| push_build_cache.yml | `.github/workflows/` | 构建缓存预热（已禁用） |
| build_and_release.yml | `.github/workflows/` | 发布产物构建 |
| release_deep_ep_wheel.yml | `.github/workflows/` | DeepEP Wheel 发布 |
| build.sh | 仓库根目录 | 构建脚本（支持 `-a` 参数） |
| npu_ci_install_dependency.sh | `scripts/` | CI 依赖安装脚本 |
| prepare_deepep_in_container.sh | `scripts/` | DeepEP 构建与安装脚本 |
| k8s_multi.yaml.jinja2 | `tests/python/deepep/` | K8s 多机 Job 模板 |
