# 本地 CI 日志位置

我已通过 `tail_run.py` 把所有 PR round 的 CI 日志下载到本地的 `.trae\logs\` 目录下，**开发可以直接拿走这些文件而无需再走 GitHub 下载**。

## 目录

```
D:\debug-pr2\.trae\logs\
```

## 文件清单

| 文件 | PR round | 状态 | 大小 | 用途 |
|------|---------|------|------|------|
| `final_27061329395.txt` | round 2 | 完成 | 10.8 MB | 0.82 基线（4/10 hybrid 通过，6 个失败用例报错） |
| `final_27064148678.txt` | round 3 | 完成 | 16.4 MB | 0.75 验证（2/10 hybrid 通过，验证更差后回退） |
| `final_27114881322.txt` | round 4 | 待生成 | - | 0.82 重提（正在跑，120 秒一轮） |
| `state_27061329395.json` | round 2 | 完成 | 87 B | 元数据：状态、URL |
| `state_27064148678.json` | round 3 | 完成 | 87 B | 元数据：状态、URL |
| `state_27114881322.json` | round 4 | 进行中 | 79 B | 元数据：状态、URL |
| `current_79874565450.txt` | - | - | 10.8 MB | 备份（与 `final_27061329395.txt` 相同） |
| `current_79881937644.txt` | - | - | 16.4 MB | 备份（与 `final_27064148678.txt` 相同） |

## 直接发给开发的具体文件

**主要发这两个**（round 2 + round 3），已经包含失败用例的完整堆栈和 server 启动日志：

- `D:\debug-pr2\.trae\logs\final_27061329395.txt` ← **重点发这个**（包含 6 个失败用例的 setUpClass ERROR + 完整 server 启动日志）
- `D:\debug-pr2\.trae\logs\final_27064148678.txt` ← 0.75 失败对照

## log 文件结构（开发拿到后怎么用）

- 0 ~ 几 MB：CI workflow 启动信息、test case 选择
- 中间段：`sglang serve` 启动命令、模型加载、HCCL 通信初始化
- 后段：unittest 跑过程，每个子用例一个 `setUpClass` / `test_xxx` / `tearDownClass` 段
- 末尾：所有 ERROR 汇总

**快速定位失败用例的 setUpClass ERROR**：
```powershell
Select-String -Path final_27061329395.txt -Pattern "ERROR: setUpClass"
```

**快速定位特定用例的 server 启动命令**：
```powershell
Select-String -Path final_27061329395.txt -Pattern "command=sglang serve" | Where-Object { $_.Line -match "deepep|ep-size 16|enable-dp-attention" }
```

**快速定位 HCCL 内存分配失败**：
```powershell
Select-String -Path final_27061329395.txt -Pattern "EL0004|HCCL"
```

## GitHub 原始链接（开发想自己再下一遍也行）

- Round 2：https://github.com/Ascend/sglang/actions/runs/27061329395
- Round 3：https://github.com/Ascend/sglang/actions/runs/27064148678
- Round 4：https://github.com/Ascend/sglang/actions/runs/27114881322
