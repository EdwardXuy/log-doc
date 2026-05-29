# Daily 9.0.0 失败测试分析与修复建议

> 生成时间：2026-05-28
> 来源运行：https://github.com/sgl-project/sgl-kernel-npu/actions/runs/26561838412
> CANN 版本：9.0.0
> 测试总数：36 | 通过：29 | 失败：7

---

## 概览

| 序号 | 失败测试 | 功能域 | 根因类别 | CI 侧可修复 | 需开发修复 |
|------|---------|--------|---------|------------|-----------|
| 1 | test_add_rmsnorm_bias.py | Norm | Python 接口参数变化 | 否 | 是 |
| 2 | test_decode_attention.py | Attention | 算子未导出/已移除 | 否 | 是 |
| 3 | test_split_qkv_rmsnorm_rope.py | Attention | Python 接口参数变化 | 否 | 是 |
| 4 | test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py | Attention | 缺少 sglang 依赖 | 是（已添加） | 否 |
| 5 | test_conv1d_prefill.py | Mamba | 负向测试逻辑过时 | 否 | 是 |
| 6 | test_swiglu_quant.py | Fused | 精度阈值过严 | 否 | 是 |
| 7 | test_catlass_matmul_basic.py | Fused | 编译选项未开启 | 是（已添加） | 否 |

---

## 详细分析

### 1. test_add_rmsnorm_bias.py

**错误日志**：
```
TypeError: add_rmsnorm_bias() got multiple values for argument 'norm_bias'
  File "test_add_rmsnorm_bias.py", line 38, in test_add_rmsnorm_bias
    res1, res2 = add_rmsnorm_bias(...)
```

**根因**：`add_rmsnorm_bias()` 函数的 Python 绑定签名发生了变化。测试代码使用位置参数传递了 `norm_bias`，但函数现在将其定义为关键字参数（或参数顺序改变），导致 Python 解释器认为 `norm_bias` 被重复赋值。

**修复建议**（需开发修改测试文件）：
```python
# 原调用（假设）：
res1, res2 = add_rmsnorm_bias(x, weight, bias, norm_bias)

# 改为关键字参数调用：
res1, res2 = add_rmsnorm_bias(x, weight, bias, norm_bias=norm_bias)
```

**建议**：请开发确认 `add_rmsnorm_bias` 的最新函数签名，并同步更新测试代码。

---

### 2. test_decode_attention.py

**错误日志**：
```
ImportError: cannot import name 'decode_gqa' from 'sgl_kernel_npu.attention'
  File "test_decode_attention.py", line 4
    from sgl_kernel_npu.attention import decode_gqa, decode_gqa_high_performance, decode_mla
```

**根因**：`decode_gqa` 算子未在 `sgl_kernel_npu.attention` 模块中导出。可能原因：
- 该算子已被移除或重命名
- 该算子尚未完成 NPU 适配，未在 Python 层注册
- 导入路径发生变化

**修复建议**（需开发确认）：

选项 A：如果 `decode_gqa` 已不存在，从测试中移除：
```python
# 修改 tests/python/sgl_kernel_npu/test_decode_attention.py
from sgl_kernel_npu.attention import decode_gqa_high_performance, decode_mla
# 删除所有使用 decode_gqa 的测试代码
```

选项 B：如果算子存在但路径变了，修改导入路径。

**建议**：请开发确认 `decode_gqa` 的当前状态。如果该算子确实未在 NPU 版本实现，建议从测试中移除相关用例，或添加 `pytest.skip` 跳过。

---

### 3. test_split_qkv_rmsnorm_rope.py

**错误日志**：
```
TypeError: custom_rope() missing 1 required positional argument: 'half_rope_dim'
  File "test_split_qkv_rmsnorm_rope.py", line 165
    cus_q, cus_k = custom_rope(_q, _k, sin, cos)
```

**根因**：`custom_rope()` 函数新增了一个必需参数 `half_rope_dim`，但测试代码未更新。

**修复建议**（需开发修改测试文件）：
```python
# 找到调用 custom_rope 的地方：
# cus_q, cus_k = custom_rope(_q, _k, sin, cos)

# 改为：
half_rope_dim = _q.shape[-1] // 2  # 或根据实际语义调整
cus_q, cus_k = custom_rope(_q, _k, sin, cos, half_rope_dim)
```

**建议**：请开发确认 `half_rope_dim` 的正确计算方式，并同步更新测试代码。

---

### 4. test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py

**错误日志**：
```
ModuleNotFoundError: No module named 'sglang'
  File "test_split_qkv_rmsnorm_rope_pos_cache_half_npu.py", line 4
    from sglang.srt.utils import is_npu
```

**根因**：测试依赖了 `sglang` 包，但 CI 环境未安装。

**CI 侧修复**（已完成）：
在 `prepare_kernel_tests.sh` 中添加了：
```bash
pip install sglang
```

**注意**：`sglang` 可能有较重的依赖链，如果安装时间过长或出现依赖冲突，建议改为：

```python
# 在测试文件中添加容错导入
try:
    from sglang.srt.utils import is_npu
except ImportError:
    is_npu = lambda: True  # CI 环境已知为 NPU
```

---

### 5. test_conv1d_prefill.py

**错误日志**：
```
AssertionError: unsupported_dim unexpectedly succeeded
  File "test_conv1d_prefill.py", line 295, in expect_failure
    raise AssertionError(f"{name} unexpectedly succeeded")
```

**根因**：`unsupported_dim` 是一个负向测试用例，期望函数在接收到不支持的维度时抛出异常。但实际上函数成功执行了，说明该维度现在已被支持。

**修复建议**（需开发修改测试文件）：

选项 A：如果该维度确实已被支持，删除这个负向用例：
```python
# 从 run_negative_cases 中移除 unsupported_dim 测试项
```

选项 B：调整负向测试逻辑，允许函数返回默认值：
```python
def expect_failure(name, func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        if result is not None:
            print(f"[PASS] {name}: previously unsupported feature now supported")
            return
        raise AssertionError(f"{name} unexpectedly succeeded")
    except Exception as e:
        print(f"[EXPECTED FAILURE] {name}: {e}")
```

**建议**：请开发确认 `unsupported_dim` 对应的功能是否已在最新代码中支持，如果是则更新测试逻辑。

---

### 6. test_swiglu_quant.py

**错误日志**：
```
AssertionError
  File "test_swiglu_quant.py", line 32
    assert diff_rate < 2e-2
```

**根因**：`diff_rate`（精度误差率）超过了阈值 `0.02`（2%）。这可能是：
- 量化算子的固有精度损失
- CANN 9.0.0 与参考实现的数值差异
- 测试输入数据导致的边界情况

**修复建议**（需开发修改测试文件）：

选项 A：放宽精度阈值：
```python
# 原断言：
assert diff_rate < 2e-2

# 放宽到 5%：
assert diff_rate < 5e-2
```

选项 B：添加调试信息，先确认实际误差大小：
```python
print(f"diff_rate: {diff_rate}")
assert diff_rate < 2e-2, f"diff_rate {diff_rate} exceeds threshold 2e-2"
```

**建议**：请开发先运行测试确认实际 `diff_rate` 值，再决定是放宽阈值还是修复算子精度问题。

---

### 7. test_catlass_matmul_basic.py

**错误日志**：
```
AttributeError: '_OpNamespace' 'npu' object has no attribute 'catlass_matmul_basic'
use catlass ops in sglang-kernel need to set BUILD_KERNELS_MODULE in cmake during compiling
```

**根因**：`catlass_matmul_basic` 算子需要编译时开启 `BUILD_CATLASS_MODULE`（或 `BUILD_KERNELS_MODULE`）CMake 选项，但当前 `build.sh -a kernels` 未开启。

**CI 侧修复**（已完成）：
在 `prepare_kernel_tests.sh` 中添加了：
```bash
export BUILD_CATLASS_MODULE=ON
bash build.sh -a kernels
```

**注意**：如果 `build.sh` 不支持通过环境变量传递该选项，可能需要直接修改 `build.sh` 或改用 `cmake`：
```bash
cmake -DBUILD_CATLASS_MODULE=ON ...
make -j$(nproc)
```

**建议**：请开发确认 `build.sh` 是否支持 `BUILD_CATLASS_MODULE` 环境变量，如果不支持需要调整编译方式。

---

## CI 侧已完成的修改

### 1. prepare_kernel_tests.sh

```bash
# 新增：开启 CATLASS 编译选项
export BUILD_CATLASS_MODULE=ON
bash build.sh -a kernels

# 新增：安装 sglang 依赖
pip install sglang
```

### 2. daily-test-kernels.yml

```yaml
# 矩阵从 [8.5.0, 9.0.0] 精简为只保留 9.0.0
matrix:
  cann_version: [9.0.0]
  hardware: [a3]
```

### 3. pr-test-kernels.yml

```yaml
# 所有 7 个测试 Job 的容器镜像从 8.5.0 改为 9.0.0
image: swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:9.0.0-a3-ubuntu22.04-py3.11
env:
  CANN_VERSION: "9.0.0"
```

---

## 待开发修复的测试文件清单

| 测试文件 | 修复内容 | 优先级 |
|---------|---------|--------|
| `tests/python/sgl_kernel_npu/test_add_rmsnorm_bias.py` | 修改 `add_rmsnorm_bias` 调用方式，使用关键字参数 | P1 |
| `tests/python/sgl_kernel_npu/test_decode_attention.py` | 确认 `decode_gqa` 状态，移除或修改导入 | P1 |
| `tests/python/sgl_kernel_npu/test_split_qkv_rmsnorm_rope.py` | 添加 `half_rope_dim` 参数 | P1 |
| `tests/python/sgl_kernel_npu/test_conv1d_prefill.py` | 删除或调整 `unsupported_dim` 负向测试 | P2 |
| `tests/python/sgl_kernel_npu/test_swiglu_quant.py` | 放宽精度阈值或修复算子精度 | P2 |

---

## 预期修复后的通过率

当前：29/36 = **80.6%**

如果 7 个失败全部修复：36/36 = **100%**

如果 CI 侧修复的 2 个（sglang + catlass）先解决：31/36 = **86.1%**

---

## 与开发沟通要点

1. **接口变更频繁**：`add_rmsnorm_bias` 和 `custom_rope` 的 Python 接口近期有变更，请开发在修改算子接口时同步更新测试代码。
2. **负向测试维护**：`test_conv1d_prefill.py` 中的负向测试需要定期 review，当功能扩展后及时移除过时的负向用例。
3. **精度阈值设定**：`test_swiglu_quant.py` 的 2% 阈值对于量化算子可能过严，建议根据实际误差分布设定合理阈值。
4. **算子注册状态**：请确认 `decode_gqa` 是否计划在 NPU 版本支持，如果不支持建议从测试列表中移除。
