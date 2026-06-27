# EPD NPU 移植用例 Debug 求助报告

## 一、基本信息

| 项 | 链接/值 |
|---|---|
| **PR链接** | https://github.com/Ascend/sglang/pull/872 |
| **分支** | `EdwardXuy/sglang-Ascend:epd-npu-test` → `Ascend/sglang:testcases` |
| **GPU原始用例** | https://github.com/sgl-project/sglang/blob/main/test/registered/disaggregation/test_epd_disaggregation.py |
| **移植用例路径** | `test/registered/ascend/basic_function/EPD/test_npu_epd_disaggregation.py` |
| **CI Workflow** | `.github/workflows/single-test-npu.yml` |
| **Runner** | `linux-aarch64-a3-8` (8张A3 NPU) |
| **镜像** | `cann9.0.0-a3-20260622` |
| **测试模型** | `Qwen/Qwen2.5-VL-3B-Instruct` |
| **最新CI链接** | https://github.com/Ascend/sglang/actions/runs/28281641875/job/83798408448?pr=872 |

---

## 二、移植背景与用例设计思路

### 2.1 移植目标

将GPU版本的EPD（Encode Prefill Disaggregation）测试用例移植到NPU平台，验证VLM视觉编码器与语言模型分离部署在NPU上的正确性。

### 2.2 GPU用例测试内容

GPU版本包含6个测试类：

| 测试类 | 测试内容 | CI运行 |
|---|---|---|
| `TestEPDDisaggregationOmni` | 含image/video/mixed输入的端到端测试 | 跳过（本地运行） |
| `TestEPDDisaggregationOneEncoder` | 单encoder + prefill/decode分离，MMMU评估 | 跳过（减少运行时间） |
| `TestEPDDisaggregationQwen35` | Qwen3.5 VL模型，含video测试 | 跳过（本地运行） |
| `TestEPDDisaggregationMultiEncoders` | 双encoder + prefill/decode分离，MMMU评估 | **运行** |
| `TestEPDDisaggregationGrpcEncoderMMMU` | gRPC模式encoder，MMMU评估 | 跳过 |
| `TestEPDDisaggregationGrpcEncoderOnly` | gRPC模式encoder-only | 跳过 |

### 2.3 移植策略

根据具体要求，以下测试类**不移植**：
- `TestEPDDisaggregationGrpcEncoderMMMU` — NPU不支持`--grpc-mode`
- `TestEPDDisaggregationGrpcEncoderOnly` — NPU不支持`--grpc-mode`
- `TestEPDDisaggregationMooncake` — NPU不支持mooncake传输后端

**移植的测试类**（4个）：
1. `TestNpuEPDDisaggregationOmni`
2. `TestNpuEPDDisaggregationOneEncoder`
3. `TestNpuEPDDisaggregationQwen35`
4. `TestNpuEPDDisaggregationMultiEncoders`（CI唯一运行的测试）

### 2.4 NPU适配主要改动

| 改动项 | GPU版本 | NPU版本 | 说明 |
|---|---|---|---|
| 张量并行参数 | `--tp` | `--tp-size` | NPU后端使用不同的参数名 |
| PD传输后端 | `mooncake` | `ascend` | NPU使用ascend PD传输后端 |
| encoder传输后端 | `zmq_to_scheduler` | `zmq_to_scheduler` | NPU支持，保持不变 |
| NPU环境变量 | 无 | `ASCEND_MF_STORE_URL=tcp://127.0.0.1:24666` | Ascend Transfer Engine需要 |
| tp_size | 1 | 2 | 单个NPU算力/显存不足，使用tp=2 |
| 总用卡数 | 4卡H100 | 8卡A3 | encode1(0-1)+encode2(2-3)+prefill(4-5)+decode(6-7) |
| server_type | 多种 | `server` | NPU只支持standard server模式 |
| grpc模式 | 支持 | 不支持 | 跳过相关测试类 |
| mm-global-cache | 支持 | 不支持 | 跳过相关测试 |
| mem-fraction-static | 默认(0.9) | 0.5 | NPU显存预分配过大导致OOM |

---

## 三、历次CI运行失败分析

### 3.1 第一次失败：ASCEND_MF_STORE_URL 缺失

**CI链接**: https://github.com/Ascend/sglang/actions/runs/28175349433/job/83450092770?pr=872

**错误日志**:
```
ERROR [SMEM smem.cpp:40 smem_create_config_store] input store URL is null.
```

**分析**: Ascend Transfer Engine (SMEM) 初始化时需要 `ASCEND_MF_STORE_URL` 环境变量，测试启动服务器时未设置。

**已修复**: 在测试文件中添加 `NPU_ENV` 字典，包含 `ASCEND_MF_STORE_URL=tcp://127.0.0.1:24666`，所有 `popen_launch_server` 调用传入 `env=NPU_ENV`。

---

### 3.2 第二次失败：--tp 参数未识别

**CI链接**: https://github.com/Ascend/sglang/actions/runs/28212777154/job/83577473211?pr=872

**错误日志**:
```
unrecognized arguments: --tp 2
```

**分析**: GPU版本使用 `--tp` 参数指定张量并行大小，但NPU后端使用 `--tp-size` 参数。

**已修复**: 将测试文件中所有10处 `--tp` 替换为 `--tp-size`（encoder 4处、prefill 3处、decode 3处）。

---

### 3.3 第三次失败：lmms_eval 模块缺失

**CI链接**: https://github.com/Ascend/sglang/actions/runs/28214835390/job/83583591809?pr=872

**错误日志**:
```
/usr/local/python3.11.15/bin/python3: No module named lmms_eval
```

**分析**: CI环境未安装lmms-eval依赖。MMMU评估需要通过 `python3 -m lmms_eval` 命令执行。

**已修复**: 在workflow中添加lmms-eval安装步骤，clone v0.3.3源码并后台安装，同时设置PYTHONPATH作为兜底。

---

### 3.4 第四次失败：decord 在aarch64上不可用

**CI链接**: https://github.com/Ascend/sglang/actions/runs/28214512403/job/83582584609?pr=872

**错误日志**:
```
ERROR: Could not find a version that satisfies the requirement decord
```

**分析**: lmms-eval依赖decord（视频解码库），但PyPI上没有aarch64/arm64架构的wheel包。CI runner是ARM架构。

**已修复**: 参考 `full-test-npu.yml` 的模式，改为：
1. 预安装核心依赖（datasets, evaluate, sentence_transformers等）
2. clone lmms-eval源码
3. 后台安装（失败不阻塞）
4. PYTHONPATH设置源码目录

---

### 3.5 第五次失败：pytz 等核心依赖缺失

**CI链接**: （与第四次同一轮的后续错误）

**错误日志**:
```
ModuleNotFoundError: No module named 'pytz'
```

**分析**: lmms_eval 导入时需要多个核心依赖（pytz, jsonlines, numexpr等），后台安装还没完成就开始跑测试了。

**已修复**: 增加预安装的依赖包列表，确保导入lmms_eval所需的核心包都已安装。

---

### 3.6 第六次失败：pip下载网络中断

**CI链接**: https://github.com/Ascend/sglang/actions/runs/28222216480/job/83605839783?pr=872

**错误日志**:
```
pip._vendor.urllib3.exceptions.ProtocolError: ('Connection broken: IncompleteRead
(77934571 bytes read, 7241132 more expected)', IncompleteRead(...))
```

**分析**: 安装了20+个依赖包，从默认PyPI源下载速度慢且网络不稳定，大文件下载时连接中断。

**已修复**: 
1. 切换到华为云PyPI镜像源（与runner同机房）
2. 添加 `--retries 10 --timeout 300` 重试机制

---

### 3.7 第七次失败：accelerate/sacrebleu 等依赖缺失

**CI链接**: https://github.com/Ascend/sglang/actions/runs/28276244129/job/83783319998?pr=872
后续还有几轮类似的依赖缺失问题。

**错误日志**:
```
ModuleNotFoundError: No module named 'accelerate'
ModuleNotFoundError: No module named 'sacrebleu'
...
```

**分析**: lmms_eval 依赖项多，逐步发现一个个缺失。

**已修复**: 逐步补全依赖列表，参考 `full-test-npu.yml` 的完整依赖列表。

---

### 3.8 第八次失败：NPU OOM（显存不足）

**CI链接**: https://github.com/Ascend/sglang/actions/runs/28279180958/job/83791537596?pr=872

**错误日志**:
```
torch_npu.memory: [ERROR] [4744] NPU out of memory.
Tried to allocate 22.81 GiB (NPU 5; 61.28 GiB total capacity; 53.64 GiB already allocated)
```

**关键发现**:
- 3B模型TP=2，每卡模型权重仅约3GB，但已分配53.64GB
- 尝试分配22.81GB（固定大小，与batch_size无关）
- batch_size从64降到8后，OOM大小完全一样

**分析**: 不是batch_size问题，而是 `mem-fraction-static=0.8` 导致KV cache预分配过大。3B模型本身很小，但KV cache按整张卡的80%预分配，导致OOM。

**已修复**: 将 `--mem-fraction-static` 从 0.8 降到 0.5。
（参考：其他NPU VLM测试 `vlm_utils.py` 中用的是 `0.35`）

---

### 3.9 第九次失败（当前）：PD分离 KVTransferError

**CI链接**: https://github.com/Ascend/sglang/actions/runs/28281641875/job/83798408448?pr=872

**当前状态**: ⚠️ **服务器启动成功，但MMMU评估部分请求失败，准确率不足**

**服务器启动成功**:
```
[TP0] Load weight begin. avail mem=60.87 GB
[TP1] Load weight begin. avail mem=61.11 GB
Multimodal attention backend not set. Use sdpa.
EPD MMReceiver: using transport_mode=http
INFO: Uvicorn running on http://127.0.0.1:11200
The server is fired up and ready to roll!
```

**错误日志**:
```
Prefill transfer failed for request ... with exception KVTransferError(bootstrap_room=...):
  Decode instance could be dead, remote mooncake session 10.0.0.130:20635_4738 is not alive

Decode transfer failed for request ... with exception KVTransferError(bootstrap_room=...):
  Failed to get kvcache from prefill instance, it might be dead

POST /v1/chat/completions HTTP/1.1" 500 Internal Server Error
```

**测试结果**:
- 总样本数: 50
- 准确率: **0.26** (预期 ≥ 0.40)
- 大量请求（约50%）返回 Internal Server Error
- 失败的请求被lmms_eval计为错误，拉低了准确率

**已尝试修复**（最新commit，待CI验证）:
- 将 lmms_eval 的 `--batch_size` 从 64 降到 4
- 推测：高并发（batch_size=64）导致PD分离的prefill/decode服务器过载，KV传输连接超时断开

---

## 四、当前状态

### 4.1 已解决的问题 ✅

| 问题 | 状态 |
|---|---|
| ASCEND_MF_STORE_URL 缺失 | ✅ 已修复 |
| --tp 参数未识别 | ✅ 已修复 |
| lmms_eval 及依赖缺失 | ✅ 已修复 |
| 网络下载不稳定 | ✅ 已修复（华为云镜像源+重试） |
| NPU OOM（显存不足） | ✅ 已修复（mem-fraction-static=0.5） |
| 服务器启动失败 | ✅ 已解决，服务器能正常启动 |

### 4.2 当前阻塞问题 ⚠️

**PD分离（Prefill-Decode disaggregation）的KV传输不稳定**

- 服务器都能正常启动，健康检查通过
- 低并发请求可能正常，但高并发（batch_size=64）下大量请求失败
- 错误：`KVTransferError: remote mooncake session is not alive`
- 最终MMMU准确率只有0.26（预期≥0.40），大量请求失败拉低了准确率

### 4.3 一个重要观察

日志里写的是 `mooncake session`，但我们在 `NpuEPDBase.setUpClass` 中设置的是：
```python
cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
```

为什么ascend后端的错误日志里会提到mooncake？是底层实现复用了mooncake的代码，还是参数没有生效？

---

## 五、猜测与最可能的原因分析

> ⚠️ 以下是猜测，不是最终结论，供开发参考

### 猜测1：PD分离 ascend 传输后端在高并发下不稳定（最可能）

**可能性：高**

- batch_size=64时大量请求失败，batch_size=4可能会好一些（待验证）
- 错误表现为连接断开/会话失效，像是超时或过载导致的
- 服务器启动和健康检查都正常，说明基础功能是通的

**需要确认**:
- ascend PD传输后端的并发能力如何？
- 有没有超时参数可以调大？
- 高并发下有没有已知的稳定性问题？

---

### 猜测2：8卡NPU之间的PD传输需要特殊配置

**可能性：中高**

当前NPU分配：
- encode1: NPU 0-1
- encode2: NPU 2-3
- prefill: NPU 4-5
- decode: NPU 6-7

PD分离（prefill↔decode）需要在NPU 4-5和6-7之间传输KV cache。

**问题**:
- 同机8卡之间的PD传输用什么通道？
- 是否需要配置RDMA设备？`--disaggregation-ib-device` 需要设置吗？
- `test_npu_multi_node_utils.py` 里有 `--disaggregation-transfer-backend ascend`，但那是多节点的，单节点8卡需要什么配置？

**参考**: 现有 `test_npu_disaggregated_vlm.py` 只有EPD分离（encoder+language），没有PD分离（prefill+decode），所以没有参考。

---

### 猜测3：mooncake 日志说明参数可能没生效

**可能性：中**

错误日志里写的是 `remote mooncake session ... is not alive`，但我们设置的是 `--disaggregation-transfer-backend ascend`。

可能的原因：
1. ascend后端底层复用了mooncake的代码/日志，名字还是叫mooncake
2. 参数没有正确传入，实际还是用的mooncake
3. 某些路径下默认fallback到mooncake

**验证方法**: 检查prefill和decode进程的启动参数，确认`--disaggregation-transfer-backend`是否正确设置。

---

### 猜测4：NPU环境的PD分离需要额外的环境变量或配置

**可能性：中**

GPU的PD分离（mooncake后端）需要：
- RDMA设备配置
- 特定的内核模块
- 大页内存等

NPU的ascend后端可能也需要类似的东西，但我们没配置：
- 有没有 `ASCEND_*` 环境变量需要设置？
- 有没有设备需要通过 `--disaggregation-ib-device` 指定？

---

### 猜测5：mm-global-cache 和 EPD 的交互问题

**可能性：低**

MultiEncoders测试开启了 `--enable-prefix-mm-cache`，这在EPD+PD分离的场景下可能有问题。

不过OneEncoder测试也有同样的配置，只是CI没跑。

---

## 六、建议排查方向

### 6.1 快速验证（优先）

1. **确认 ascend PD 传输后端参数是否生效**
   - 查看prefill/decode进程的完整启动参数
   - 确认 `--disaggregation-transfer-backend ascend` 确实传进去了

2. **手动发低并发请求测试**
   - 不用lmms_eval，直接用curl发1-2个多模态请求
   - 看PD分离是否正常工作
   - 逐步增加并发，找到临界点

3. **检查各服务器日志**
   - prefill服务器有没有收到KV传输请求？
   - decode服务器有没有正常接收KV cache？
   - 两边的错误日志各是什么？

### 6.2 深入排查

1. **对比GPU和NPU的PD分离实现**
   - GPU用mooncake，NPU用ascend
   - ascend后端的实现位置在哪里？
   - 错误日志里的"mooncake session"是怎么回事？

2. **检查NPU PD传输的配置要求**
   - 单节点8卡需要配置RDMA设备吗？
   - `--disaggregation-ib-device` 需要设置吗？
   - 有没有其他必需的环境变量？

3. **参考多节点NPU测试**
   - `test_npu_multi_node_utils.py` 里有PD分离的配置
   - 虽然是多节点，但单节点是否可以参考？

### 6.3 简化问题

如果复杂场景难排查，可以逐步简化：
1. 先跑 OneEncoder 测试（单encoder，PD分离）
2. 先不用encoder分离，只测PD分离（prefill+decode）
3. 先跑小batch、单图片
4. 确认基础PD分离链路通了再上EPD+MMMU

---

## 七、关键代码位置

| 文件 | 说明 |
|---|---|
| `test/registered/ascend/basic_function/EPD/test_npu_epd_disaggregation.py` | 移植的NPU测试用例 |
| `test/registered/disaggregation/test_epd_disaggregation.py` | GPU原始用例（参考） |
| `test/registered/ascend/basic_function/EPD/test_npu_disaggregated_vlm.py` | 现有NPU EPD测试（只有EPD，没有PD分离） |
| `python/sglang/test/ascend/disaggregation_utils.py` | NPU PD分离基础类（TestDisaggregationBase） |
| `python/sglang/test/ascend/e2e/test_npu_multi_node_utils.py` | 多节点NPU测试（有PD分离配置参考） |
| `.github/workflows/single-test-npu.yml` | CI流水线配置 |
| `python/sglang/test/kits/mmmu_vlm_kit.py` | MMMU评估工具类 |
| `python/sglang/srt/disaggregation/` | EPD/PD核心实现（需关注NPU适配部分） |

---

## 八、环境信息

- **架构**: linux-aarch64 (ARM)
- **NPU**: 8 × Ascend A3 (每卡约61GB显存)
- **CANN版本**: 9.0.0 (镜像: cann9.0.0-a3-20260622)
- **Python**: 3.11.15
- **PyTorch**: NPU版本（镜像内置）
- **lmms-eval版本**: v0.3.3
- **数据集**: MMMU val（limit=50）
- **当前batch_size**: 4（最新修改，待验证）
- **mem-fraction-static**: 0.5
- **PD传输后端**: ascend（已设置，但日志里有mooncake字样）
- **encoder传输后端**: zmq_to_scheduler
