# Qwen3-VL-4B EPD NPU 问题分析报告

## 一、问题概述

**核心问题**：同样的EPD（Encode-Prefill-Decode）分离测试脚本，更换模型从Qwen2.5-VL-3B到Qwen3-VL-4B后，测试表面通过但实际所有请求均失败，准确率为随机水平。

**现象**：

- CI测试显示"通过"，输出准确率 0.26（阈值 0.25）
- 实际日志中所有50道MMMU测试题的5次重试全部失败
- NPU底层出现大量HYBM/SMEM通信错误
- KV缓存传输完全失败（KVTransferError）

***

## 二、相关链接

### 2.1 PR与代码链接

| 项目                 | 链接                                                                                                          | 说明                         |
| ------------------ | ----------------------------------------------------------------------------------------------------------- | -------------------------- |
| **PR #872**        | <https://github.com/Ascend/sglang/pull/872>                                                                 | EPD NPU移植PR                |
| **GPU原始用例**        | <https://github.com/Ascend/sglang/blob/testcases/test/registered/disaggregation/test_epd_disaggregation.py> | GPU版本EPD测试用例               |
| **NPU测试用例**        | `test/registered/ascend/basic_function/EPD/test_npu_epd_disaggregation.py`                                  | NPU版本EPD测试用例（Qwen3-VL-4B）  |
| **Qwen2.5-VL EPD成功用例** | `test/registered/ascend/basic_function/EPD/test_npu_epd_disaggregation.py`（Qwen2.5-VL版本） | Qwen2.5-VL-3B EPD测试（同脚本不同模型） |
| **Ascend传输引擎**     | `python/sglang/srt/disaggregation/ascend/transfer_engine.py`                                                | NPU MemFabric传输引擎实现        |
| **Ascend连接管理**     | `python/sglang/srt/disaggregation/ascend/conn.py`                                                           | NPU KV传输连接实现               |
| **移植报告**           | `EPD_NPU_移植报告.md`                                                                                           | GPU→NPU移植详细记录              |

### 2.2 CI日志链接

| 测试场景                      | CI日志链接                                                                             | 状态         | 说明                 |
| ------------------------- | ---------------------------------------------------------------------------------- | ---------- | ------------------ |
| **Qwen3-VL-4B EPD（问题日志）** | <https://github.com/Ascend/sglang/actions/runs/28506733142/job/84496878893?pr=872> | ❌ 表面通过实际失败 | 所有请求失败，准确率0.26为随机值 |
| **Qwen3-VL-4B EPD（最新日志）** | <https://github.com/Ascend/sglang/actions/runs/28493754164/job/84455643141?pr=872> | ❌ 表面通过实际失败 | 与上述问题一致            |
| **Qwen2.5-VL-3B EPD**     | <https://github.com/Ascend/sglang/actions/runs/28368636789/job/84040641568?pr=872> | ✅ 成功       | 模型正常运行             |

### 2.3 模型与论文链接

| 项目                   | 链接                                             | 说明              |
| -------------------- | ---------------------------------------------- | --------------- |
| **Qwen3-VL-4B论文/介绍** | <https://lmmarketcap.com/upcoming/qwen3-vl-4b> | Qwen3-VL-4B模型介绍 |
| **MMMU基准**           | <https://arxiv.org/pdf/2511.21631>             | MMMU多模态理解基准     |

***

## 三、HYBM/SMEM底层通信错误分析

### 3.1 错误统计

在CI日志中，共发现 **46+ 处** NPU底层通信相关错误：

| 错误类型                           | 出现次数 | 错误码    | 严重程度  |
| ------------------------------ | ---- | ------ | ----- |
| `HYBM IpcOpenMemory failed`    | 8次   | 507899 | 🔴 严重 |
| `SMEM session not found`       | 10+次 | -2000  | 🔴 严重 |
| `BatchTransferSyncWrite error` | 2次   | -2000  | 🔴 严重 |
| `import new slices failed`     | 8次   | -6     | 🟠 高  |
| `KVTransferError`              | 193次 | -      | 🔴 致命 |

### 3.2 错误详细分析

#### 3.2.1 HYBM IpcOpenMemory 失败

**日志片段**（`ci-log-pr872-latest.txt:1158`）：

```
ERROR [HYBM hybm_dev_user_legacy_segment.cpp:394 ImportSliceInfo] 
IpcOpenMemory(000b419b0000000000030b5d0b000855000b0000000000000000000000000000) failed:507899,
sdid=140247054, pid=737690, deviceId=6, sliceInfo.logicDeviceId=5
```

**分析**：

- **HYBM**（HBM Memory Management）：负责NPU HBM内存管理和跨进程共享
- `IpcOpenMemory` 失败表示进程间内存共享（IPC）失败
- 错误码 `507899` 是Ascend驱动的内部错误码
- **关键现象**：`deviceId=6` 但 `logicDeviceId=5`，物理设备ID与逻辑设备ID不匹配
- 发生在Prefill和Decode服务器建立KV缓存共享内存时

**可能原因**：

1. **MemFabric配置不正确**：`ASCEND_MF_STORE_URL` 环境变量设置的存储地址无法正确建立跨进程内存映射
2. **模型架构差异导致的KV缓存结构不同**：Qwen3-VL使用SigLIP-2 + DeepStack新架构，KV缓存布局与Qwen2.5-VL不同，可能导致MemFabric内存注册和设备ID映射异常
3. **逻辑设备ID与物理设备ID不匹配**：EPD架构下Prefill和Decode服务器各自分配设备ID，导致跨服务器内存共享时设备ID映射错误

#### 3.2.2 SMEM Session Not Found

**日志片段**（`ci-log-pr872-latest.txt:1330`）：

```
ERROR [SMEM smem_trans_entry.cpp:236 BatchSyncTransfer] 
session:(10.0.1.11:19347_6018)(0b:01:00:0a:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:93:4b:82:17) not found.

ERROR [ADAPTER pytransfer.cpp:136 BatchTransferSyncWrite] 
SMEM API smem_trans_batch_write happen error, ret=-2000
```

**分析**：

- **SMEM**（Shared Memory）：负责跨设备共享内存传输
- `session not found` 表示传输会话不存在或已失效
- 错误码 `-2000` 表示会话未找到
- **关键现象**：Prefill端尝试发送KV缓存到Decode端时，Decode端的session不存在
- 紧接着出现 `KVTransferError`，确认KV传输失败

**可能原因**：

1. **MemFabric会话初始化失败**：Decode服务器的AscendTransferEngine初始化时未能正确注册到store\_url
2. **会话超时或自动清理**：Prefill服务器处理时间较长，导致Decode端会话超时被清理
3. **网络配置问题**：虽然是本机回环（127.0.0.1），但MemFabric使用TCP端口通信，可能存在端口冲突或防火墙问题

#### 3.2.3 KVTransferError 级联失败

**日志片段**（`ci-log-pr872-latest.txt:1335`）：

```
[TP0] Decode transfer failed for request ... with exception KVTransferError(bootstrap_room=5717279478565220750): 
Failed to get kvcache from prefill instance, it might be dead

[TP0] Prefill transfer failed for request ... with exception KVTransferError(bootstrap_room=5717279478565220750): 
Failed to send kv chunk of 5717279478565220750 to 10.0.1.11:33137
```

**分析**：

- KV传输失败是HYBM/SMEM错误的直接结果
- 日志中出现 **193次** KVTransferError，意味着几乎所有请求的KV传输都失败
- 错误同时出现在Prefill→Decode和Decode→Prefill两个方向
- 触发熔断器（Circuit Breaker）机制，服务进入 `open` 状态，后续请求直接返回503

### 3.3 NPU配置问题根因假设

按可能性排序：

| # | 根因假设 | 依据 | 影响范围 |
| - | - | - | - |
| 1 | **Qwen3-VL模型架构差异导致KV传输不兼容** | 同一套EPD脚本+TP=2：Qwen2.5-VL-3B 成功，Qwen3-VL-4B 失败；Qwen3-VL使用不同的视觉编码器（SigLIP-2）、DeepStack等架构，可能导致MemFabric内存注册和KV传输格式不匹配 | Qwen3-VL系列模型 |
| 2 | **模型更大导致显存压力，KV缓存注册失败** | 4B模型比3B占用更多显存，`batch_register_memory` 静默失败（日志为debug级别），后续传输时找不到内存 | 大模型EPD场景 |
| 3 | **`ASCEND_MF_STORE_URL` 未正确传递到所有进程** | Prefill和Decode服务器各自独立启动，环境变量可能未正确设置或传递不完整 | 所有EPD场景 |
| 4 | **AscendTransferEngine初始化顺序问题** | 日志显示HYBM错误发生在服务启动早期，可能是HCCL初始化与MemFabric初始化冲突 | 特定模型配置 |
| 5 | **CANN驱动版本与Qwen3-VL所需算子不兼容** | 使用 `cann9.0.0-a3-20260611` 镜像，Qwen3-VL的新算子可能存在驱动兼容问题 | Qwen3-VL系列模型 |

***

## 四、准确率低的原因分析（按影响程度排序）

### 原因1：所有请求完全失败，准确率为随机猜测水平（影响程度：🔴 100%）

**现象**：

- 50道MMMU题，每道题重试5次，全部失败（`All 5 attempts failed`）
- 最终输出准确率 0.26（50题中约13题"正确"）
- MMMU为4选1选择题，随机猜测准确率约 0.25

**分析**：

- lmms-eval框架在请求完全失败时，对失败样本的处理导致了"假准确率"
- 可能的处理方式：失败请求被标记为错误答案，或使用默认/空输出进行匹配
- 0.26 ≈ 0.25（随机水平），说明没有任何真实推理发生
- **这是最核心的问题：不是准确率低，而是服务完全不可用**

### 原因2：NPU底层通信完全失效（HYBM/SMEM错误）（影响程度：🔴 100%）

**现象**：

- Prefill与Decode服务器间KV缓存传输100%失败
- 大量HYBM IpcOpenMemory和SMEM session not found错误
- 熔断器被触发，服务进入不可用状态

**分析**：

- EPD架构依赖Prefill和Decode服务器间的KV缓存传输
- 如果KV传输失败，整个推理流程无法完成
- 这是导致原因1（所有请求失败）的技术根源

---

### 假设服务正常时，准确率仍低于论文值的原因分析

即使解决了上述NPU通信问题，服务完全正常运行，当前评估配置下的准确率仍会远低于论文值（约59.5%）。以下是按影响程度排序的5个原因：

#### 🔴 原因 3：openai_compatible 模式，不是 Qwen3-VL 原生适配

这是最大的原因。看日志里的配置：

```
model: 'openai_compatible',
model_args: 'model_version="/root/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct",tp=1'
```

我们用的是 lmms-eval 的通用 `openai_compatible` 模式，而不是 Qwen3-VL 的**原生适配**（`qwen3_vl` 模型类型）。区别在于：

| 维度 | openai_compatible（我们用的） | qwen3_vl 原生适配 |
|---|---|---|
| Chat Template | 通用格式，可能不对 | 官方正确格式 |
| 图像预处理 | 通用处理 | Qwen3-VL 专用（SigLIP-2 归一化、DeepStack 等） |
| 图像 token 格式 | 通用 `<image>` 占位 | Qwen3-VL 专用的视觉 token 排列 |
| 特殊 token 处理 | 不支持 | 完整支持 |

这一项估计就能导致 **10-20% 的准确率差距**。

#### 🟠 原因 4：只跑 50 道题，抽样方差极大

只抽了 50 道题，样本量太小：

- 900 道题覆盖 30 个学科，50 道题可能某些学科完全没抽到
- 运气成分很大，抽题难度分布不稳定
- 正常抽样误差 ±10% 都很正常

#### 🟡 原因 5：Zero-shot，没有 CoT 推理

看日志里的 prompt：

```
multiple_choice_prompt: "Answer with the option's letter from the given choices directly."
```

就是让模型**直接输出答案字母**，没有任何推理过程。而论文/官方评测通常会用：

- Chain-of-Thought（CoT）推理
- 精心设计的 system prompt
- Few-shot 示例

CoT 对 MMMU 这种需要推理的 benchmark 提升很大，通常能有 **5-15% 的提升**。

#### 🟡 原因 6：max_new_tokens: 16 太短了

```
generation_kwargs: {max_new_tokens: 16, until: ['\n\n']}
```

只给模型 16 个 token 的输出空间。虽然理论上只需要输出 "A"、"B" 等单个字母，但：

- 有些模型可能会先说 "The answer is A." 然后再输出答案
- 输出太短可能导致格式不对，解析失败

#### 🟢 原因 7：EPD 架构的精度损耗（相对较小）

EPD 分布式架构（encoder + prefill + decode）：

- KV cache 跨进程传输，有序列化/反序列化
- 多段式推理的累积误差

但这个影响应该是最小的，估计在 **1-3%** 左右。

***

## 五、为什么同样的EPD脚本更换模型后跑不通？对比分析

### 5.1 背景说明

**核心问题**：在**同一个EPD测试脚本**中，使用 Qwen2.5-VL-3B 模型可以正常运行并通过测试，但更换为 Qwen3-VL-4B 后，虽然CI表面通过，但实际所有请求都失败。

两个模型使用的是同一套 EPD 分离架构代码（encoder + prefill + decode），测试脚本结构完全一致，仅仅是模型权重和相关配置。

### 5.2 Qwen2.5-VL-3B vs Qwen3-VL-4B 配置对比

| 配置项 | Qwen2.5-VL-3B EPD（成功） | Qwen3-VL-4B EPD（失败） | 差异 |
|---|---|---|---|
| **测试架构** | EPD分离架构 | EPD分离架构 | ✅ 相同 |
| **测试脚本** | 同一套EPD测试框架 | 同一套EPD测试框架 | ✅ 相同 |
| **TP大小** | TP=2 | TP=2 | ✅ 相同 |
| **KV传输** | 需要（Prefill↔Decode） | 需要（Prefill↔Decode） | ✅ 相同 |
| **MemFabric** | 使用 | 使用 | ✅ 相同 |
| **模型** | Qwen2.5-VL-3B-Instruct | Qwen3-VL-4B-Instruct | ❌ 不同 |
| **模型大小** | 3B | 4B | ❌ 不同 |
| **视觉编码器** | Qwen2.5-VL 原生 | SigLIP-2 + DeepStack（Qwen3-VL新架构） | ❌ 不同 |
| **测试结果** | ✅ 成功 | ❌ 表面通过实际失败 | ❌ 不同 |

### 5.3 核心差异点分析

既然是同一套EPD脚本、同样TP=2，为什么换个模型就跑不通了？可能的原因按可能性排序：

1. **模型架构差异（最可能）**：
   - Qwen3-VL系列使用了全新的视觉编码器（SigLIP-2 + DeepStack），与Qwen2.5-VL完全不同
   - KV缓存的布局、视觉token的排列方式、视觉特征的注入方式可能不同
   - 可能导致KV传输时的内存注册和传输格式不匹配
   - **这是最可疑的根因：Qwen3-VL的新架构与现有EPD KV传输逻辑不兼容**

2. **模型大小导致的显存压力**：
   - 4B模型比3B模型占用更多显存
   - 可能导致MemFabric可用于传输的内存不足
   - 或者内存注册失败（`batch_register_memory` 静默失败）

3. **Qwen3-VL新算子与CANN驱动兼容性**：
   - Qwen3-VL可能使用了一些新的算子或模型结构
   - `cann9.0.0-a3-20260611` 镜像的驱动版本可能不完全支持
   - 导致初始化阶段就出现问题，进而影响MemFabric通信

### 5.4 验证建议

为了精确定位问题，建议按以下顺序进行验证：

| # | 验证实验 | 目的 | 预期结果 |
|---|---|---|---|
| 1 | Qwen3-VL-4B 单卡非EPD模式 + MMMU | 验证模型本身在NPU上能否正常推理 | 准确率 > 0.4，证明模型本身没问题 |
| 2 | 对比两模型的KV缓存结构和大小 | 验证是否因模型架构差异导致传输失败 | 确认Qwen3-VL的KV缓存布局是否与Qwen2.5-VL不同 |
| 3 | Qwen3-VL-4B EPD模式 + 更小的batch/更长的启动等待 | 验证是否因显存压力或初始化时序问题 | 如果成功，说明是显存/时序问题 |
| 4 | 增加请求成功率校验 | 验证测试框架是否能正确检测失败 | 成功率应为100%或接近 |
| 5 | 检查MemFabric初始化日志（debug级别） | 验证 `batch_register_memory` 是否成功 | 确认内存注册是否静默失败 |

***

## 六、修复建议

### 6.1 优先级P0：紧急修复

1. **验证MemFabric配置**
   - 确认 `ASCEND_MF_STORE_URL` 环境变量在Prefill和Decode服务器中都正确设置
   - 检查AscendTransferEngine初始化日志，确认无错误
   - 参考 `transfer_engine.py:43` 中的store\_url读取逻辑
2. **排查Qwen3-VL模型架构与EPD KV传输的兼容性**
   - 对比Qwen2.5-VL和Qwen3-VL的KV缓存结构差异
   - 检查视觉编码器（SigLIP-2 + DeepStack）对KV缓存布局的影响
   - 验证模型初始化时的内存注册是否成功
3. **增加测试健壮性**
   - 将准确率阈值从0.25提高到至少0.35
   - 增加请求成功率检查，成功率低于90%直接判失败
   - 检查熔断器状态，服务不可用时直接失败

### 6.2 优先级P1：短期改进

1. **完善错误日志**
   - 将 `batch_register_memory` 失败从debug级别提升到warning/error
   - 增加MemFabric初始化结果的明确日志
   - KV传输失败时输出更详细的错误上下文
2. **增加配置校验**
   - 启动时检查 `ASCEND_MF_STORE_URL` 是否已设置
   - 验证EPD模式下必需的环境变量和依赖
   - TP>1时增加额外的配置检查
3. **优化重试策略**
   - 对系统性错误（如503服务不可用）快速失败，不要重试5次
   - 区分"偶发错误"和"系统性失败"

### 6.3 优先级P2：长期优化

1. **补充测试矩阵**
   - 单卡 vs 多卡
   - EPD vs 非EPD
   - 不同TP大小
   - 不同模型架构
2. **性能基准对比**
   - 建立GPU vs NPU的性能/准确率基线
   - 定期对比，及时发现回归问题
3. **文档完善**
   - 补充EPD on NPU的配置要求
   - 列出支持的模型和配置组合
   - 常见错误排查指南

***

## 七、结论

**核心结论**：Qwen3-VL-4B在EPD架构下的失败**不是TP配置问题**（两模型都是TP=2），而是**Qwen3-VL模型架构与现有EPD KV传输机制的兼容性问题**（或模型更大导致显存压力不足）。

**证据链**：

1. 启动阶段即出现HYBM/SMEM底层错误 → 内存共享初始化失败
2. KV缓存传输100%失败（193次KVTransferError） → EPD核心功能失效
3. 所有请求5次重试全部失败 → 服务完全不可用
4. 准确率0.26 ≈ 随机猜测水平 → 没有真实推理发生
5. 测试因阈值过低（0.25）而"通过" → 测试设计缺陷掩盖了问题

**下一步行动**：

1. 优先排查Qwen3-VL模型架构与EPD KV传输的兼容性问题
2. 按5.4节的验证建议逐步缩小问题范围
3. 同步修复测试阈值过低的问题，避免再次出现"假通过"
4. 考虑将评估模式从 `openai_compatible` 切换到 Qwen3-VL 原生适配，以获得更准确的评测结果

