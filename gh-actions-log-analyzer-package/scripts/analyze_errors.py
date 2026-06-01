#!/usr/bin/env python3
"""GitHub Actions Log Analyzer - Error Log Analyzer

Analyzes failed job logs to classify errors and determine root causes.
Outputs: failed_tests_detail.csv + error_analysis_report.md (Chinese)
"""
import os, re, csv, json, argparse
from collections import defaultdict
from datetime import datetime

# Error classification patterns (category, sub_category, pattern_regex, description_cn)
ERROR_PATTERNS = [
    ("产品问题", "NPU算子错误", r"aclnn\w+\s+failed|call aclnn\w+ failed|EZ9999.*Inner Error|NPU function error|error code is \d+|aclnnGather failed", "NPU算子执行失败（tiling/映射/运行时错误）"),
    ("产品问题", "HCCL通信错误", r"HCCL.*error|HCCL_BUFFSIZE|communication error|rank.*timeout", "HCCL集合通信错误"),
    ("产品问题", "CANN框架错误", r"CANN.*error|E\d{5}|ASCEND.*error|Ascend.*error", "CANN框架或驱动错误"),
    ("产品问题", "Triton兼容性", r"triton\.language has no attribute|AttributeError.*triton", "Triton版本与NPU不兼容"),
    ("产品问题", "DeepEP运行时错误", r"deep_ep.*error|deep_ep_cpp|low_latency_dispatch|low_latency_combine|RuntimeError.*deep_ep", "DeepEP库运行时错误"),
    ("用例设计问题", "断言失败", r"AssertionError|assert diff.*Error|assert .*failed|Assertion.*failed", "测试断言失败（精度/行为不匹配）"),
    ("用例设计问题", "模块缺失", r"ModuleNotFoundError: No module named|ImportError: cannot import name", "Python模块缺失或导入失败"),
    ("用例设计问题", "配置属性缺失", r"Config.*has no attribute|AttributeError.*Config", "模型配置缺少必要属性"),
    ("基础设施问题", "K8s Pod调度超时", r"Pod.*Pending.*timeout|pod.*not.*ready|container.*not.*ready", "K8s Pod调度超时"),
    ("基础设施问题", "K8s Pod崩溃", r"Pod logs ended but target pattern was not detected|pod.*crashed|container.*exited", "K8s Pod启动后测试进程崩溃"),
    ("基础设施问题", "Runner/环境问题", r"runner lost|self-hosted runner|environment.*not.*ready|No such file or directory.*set_env", "自托管Runner或环境问题"),
    ("基础设施问题", "超时", r"timeout|timed out|deadline exceeded|context deadline", "执行超时"),
    ("基础设施问题", "内存不足", r"out of memory|OOM|Killed|exit code -9|Server process exited with code -9", "内存不足或进程被杀死"),
    ("未知", "Python异常", r"Traceback \(most recent call last\)", "未分类的Python异常"),
]

# Root cause classification rules (pattern, root_cause_cn, confidence, recommendation_cn)
ROOT_CAUSE_RULES = [
    (r"MoeDistributeDispatchV2 do tiling failed", "产品问题 - NPU算子Tiling缺陷", "高", "上报CANN/NPU团队：MoeDistributeDispatchV2 tiling失败"),
    (r"MoeDistributeCombineV2 do tiling failed", "产品问题 - NPU算子Tiling缺陷", "高", "上报CANN/NPU团队：MoeDistributeCombineV2 tiling失败"),
    (r"aclnnMoeDistributeDispatchV2 failed", "产品问题 - NPU算子运行时缺陷", "高", "上报CANN/NPU团队：aclnnMoeDistributeDispatchV2运行时失败"),
    (r"aclnnMoeDistributeCombineV2 failed", "产品问题 - NPU算子运行时缺陷", "高", "上报CANN/NPU团队：aclnnMoeDistributeCombineV2运行时失败"),
    (r"diff=.*e-0?5|diff=.*e-0?6", "用例设计问题 - 精度阈值过严", "中", "调整测试精度阈值或深入调查数值差异原因"),
    (r"AssertionError: Error: diff=", "用例设计问题 - 精度阈值过严", "中", "调整测试精度阈值或深入调查数值差异原因"),
    (r"HCCL_BUFFSIZE is too SMALL", "产品问题 - HCCL缓冲区配置", "高", "增大HCCL_BUFFSIZE环境变量值"),
    (r"ModuleNotFoundError: No module named 'deep_ep'", "用例设计问题 - 构建产物缺失", "高", "确保DeepEP wheel已构建并在测试前安装"),
    (r"No module named", "用例设计问题 - 依赖缺失", "中", "在requirements或测试环境中添加缺失的Python包"),
    (r"cannot import name", "用例设计问题 - API不兼容", "中", "检查导入路径与当前代码版本的兼容性"),
    (r"triton\.language has no attribute", "产品问题 - Triton NPU兼容性", "高", "升级Triton版本或实现NPU兼容的替代方案"),
    (r"Config.*has no attribute", "用例设计问题 - 模型配置过时", "中", "更新模型配置类，添加缺失的属性"),
    (r"Pod.*Pending", "基础设施问题 - K8s资源不足", "高", "检查K8s集群资源可用性"),
    (r"Pod logs ended but target pattern was not detected", "基础设施问题 - K8s Pod崩溃", "高", "检查Pod资源限制和测试进程稳定性"),
    (r"out of memory|OOM|Killed", "基础设施问题 - 资源耗尽", "高", "增加Pod内存限制或减少batch size"),
    (r"timeout|timed out", "基础设施问题 - 超时", "中", "增加步骤超时时间或优化测试执行时长"),
    (r"runner lost", "基础设施问题 - Runner异常", "高", "检查自托管Runner健康状态"),
]


def classify_error(error_text):
    for category, sub_category, pattern, description in ERROR_PATTERNS:
        if re.search(pattern, error_text, re.IGNORECASE):
            return category, sub_category, description
    return "未知", "未分类", "无法分类错误"


def determine_root_cause(error_text):
    for pattern, root_cause, confidence, recommendation in ROOT_CAUSE_RULES:
        if re.search(pattern, error_text, re.IGNORECASE):
            return root_cause, confidence, recommendation
    return "未知 - 需人工复核", "低", "手动复核日志以确定根因"


def extract_rich_error_detail(log_path, line_idx, lines):
    """Extract rich error detail with surrounding context."""
    # Look for Traceback block
    traceback_start = None
    for j in range(max(0, line_idx - 100), line_idx):
        if "Traceback (most recent call last)" in lines[j]:
            traceback_start = j
            break

    if traceback_start is not None:
        # Capture traceback + error line
        ctx_end = min(traceback_start + 40, len(lines))
        context = lines[traceback_start:ctx_end]
        # Find the actual error line (usually last non-empty line)
        error_lines = []
        for cl in context:
            stripped = cl.strip()
            if stripped and not stripped.startswith("File ") and not stripped.startswith("Traceback"):
                error_lines.append(stripped)
        # Get last few meaningful lines
        detail_lines = error_lines[-5:] if len(error_lines) >= 5 else error_lines
        detail = " | ".join(detail_lines)
        return detail, "".join(context)

    # No traceback, grab surrounding lines
    ctx_start = max(0, line_idx - 10)
    ctx_end = min(len(lines), line_idx + 15)
    context = lines[ctx_start:ctx_end]
    detail = lines[line_idx].strip()
    return detail, "".join(context)


def extract_errors_from_log(log_path):
    errors = []
    if not os.path.exists(log_path):
        return errors

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    failed_tests = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r"FAILED\s+(\S+test_npu\S+\.py)\s*::", line)
        if not m:
            m = re.search(r"FAILED\s+(\S+deepep\S+\.py)\s*::", line)
        if not m:
            m = re.search(r"FAILED\s+(\S+sgl_kernel_npu\S+\.py)\s*::", line)
        if m:
            test_path = m.group(1)
            test_name = os.path.basename(test_path).replace(".py", "")
            detail, context = extract_rich_error_detail(log_path, i, lines)
            failed_tests.append({
                "test_path": test_path,
                "test_name": test_name,
                "error_detail": detail,
                "context": context,
                "log_line": i + 1
            })
        i += 1

    # If no pytest FAILED markers, look for critical errors
    if not failed_tests:
        for i, line in enumerate(lines):
            if any(kw in line for kw in ["AssertionError", "RuntimeError", "ModuleNotFoundError", "ImportError", "EZ9999", "aclnn.*failed", "HCCL.*error", "out of memory", "Killed", "timeout"]):
                detail, context = extract_rich_error_detail(log_path, i, lines)
                failed_tests.append({
                    "test_path": "unknown",
                    "test_name": "unknown",
                    "error_detail": detail,
                    "context": context,
                    "log_line": i + 1
                })

    return failed_tests


def analyze_all_logs(logs_dir, reports_dir):
    all_errors = []
    job_error_map = defaultdict(list)
    run_url_map = {}  # run_info -> html_url

    # Load run URLs from all_runs.json
    runs_json_path = os.path.join(reports_dir, "all_runs.json")
    if os.path.exists(runs_json_path):
        try:
            with open(runs_json_path, "r", encoding="utf-8") as f:
                runs_data = json.load(f)
            for r in runs_data:
                run_key = f"{r.get('WorkflowName', 'unknown')}_run-{r.get('RunId', 'unknown')}"
                run_url_map[run_key] = r.get("HtmlUrl", "")
        except Exception:
            pass

    for root, dirs, files in os.walk(logs_dir):
        if "full-log.txt" in files:
            log_path = os.path.join(root, "full-log.txt")
            rel_path = os.path.relpath(root, logs_dir)
            parts = rel_path.split(os.sep)
            run_info = parts[0] if parts else "unknown"
            job_name = parts[1] if len(parts) > 1 else "unknown"

            job_info_path = os.path.join(root, "job-info.json")
            job_conclusion = "failure"
            if os.path.exists(job_info_path):
                try:
                    with open(job_info_path, "r", encoding="utf-8") as f:
                        job_info = json.load(f)
                    job_conclusion = job_info.get("conclusion", "failure")
                except Exception:
                    pass

            if job_conclusion != "failure":
                continue

            errors = extract_errors_from_log(log_path)
            for err in errors:
                category, sub_category, description = classify_error(err["context"])
                root_cause, confidence, recommendation = determine_root_cause(err["context"])

                record = {
                    "RunId": run_info,
                    "JobName": job_name,
                    "TestPath": err["test_path"],
                    "TestName": err["test_name"],
                    "ErrorDetail": err["error_detail"][:800],
                    "Category": category,
                    "SubCategory": sub_category,
                    "Description": description,
                    "RootCause": root_cause,
                    "Confidence": confidence,
                    "Recommendation": recommendation,
                    "LogLine": err["log_line"],
                    "RunUrl": run_url_map.get(run_info, ""),
                }
                all_errors.append(record)
                job_error_map[(run_info, job_name)].append(record)

    return all_errors, job_error_map


def generate_error_csv(all_errors, reports_dir):
    csv_path = os.path.join(reports_dir, "failed_tests_detail.csv")
    fieldnames = ["RunId", "JobName", "TestPath", "TestName", "ErrorDetail", "Category", "SubCategory", "Description", "RootCause", "Confidence", "Recommendation", "LogLine", "RunUrl"]
    if not all_errors:
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return csv_path

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_errors:
            writer.writerow(rec)
    return csv_path


def generate_error_report(all_errors, job_error_map, reports_dir, since_date, until_date):
    dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append("# 错误日志分析报告")
    L.append("")
    L.append(f"**生成时间**: {dt}")
    L.append(f"**分析范围**: {since_date} 至 {until_date}")
    L.append(f"**错误记录总数**: {len(all_errors)}")
    L.append("")
    L.append("---")
    L.append("")

    # 1. 错误类别汇总
    L.append("## 1. 错误类别汇总")
    L.append("")
    cat_counts = defaultdict(int)
    subcat_counts = defaultdict(int)
    for rec in all_errors:
        cat_counts[rec["Category"]] += 1
        subcat_counts[(rec["Category"], rec["SubCategory"])] += 1

    L.append("| 类别 | 数量 | 占比 |")
    L.append("|------|------|------|")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        pct_val = round(cnt / len(all_errors) * 100, 1) if all_errors else 0
        L.append(f"| {cat} | {cnt} | {pct_val}% |")
    L.append("")

    L.append("### 子类别细分")
    L.append("| 类别 | 子类别 | 数量 |")
    L.append("|------|--------|------|")
    for (cat, sub), cnt in sorted(subcat_counts.items(), key=lambda x: -x[1]):
        L.append(f"| {cat} | {sub} | {cnt} |")
    L.append("")
    L.append("---")
    L.append("")

    # 2. 根因分析
    L.append("## 2. 根因分析")
    L.append("")
    root_cause_counts = defaultdict(int)
    for rec in all_errors:
        root_cause_counts[rec["RootCause"]] += 1

    L.append("| 根因 | 数量 | 置信度 | 修复建议 |")
    L.append("|------|------|--------|----------|")
    for rc, cnt in sorted(root_cause_counts.items(), key=lambda x: -x[1]):
        rec_example = next((r for r in all_errors if r["RootCause"] == rc), None)
        recommendation = rec_example["Recommendation"] if rec_example else ""
        confidence = rec_example["Confidence"] if rec_example else ""
        L.append(f"| {rc} | {cnt} | {confidence} | {recommendation} |")
    L.append("")
    L.append("---")
    L.append("")

    # 3. 详细错误记录
    L.append("## 3. 详细错误记录")
    L.append("")
    for (run_id, job_name), errors in sorted(job_error_map.items()):
        # Extract run URL from first error
        run_url = errors[0].get("RunUrl", "") if errors else ""
        run_link = f"[{run_id}]({run_url})" if run_url else run_id
        L.append(f"### {run_link} / {job_name}")
        L.append("")
        L.append("| 测试 | 错误详情 | 类别 | 根因 | 置信度 |")
        L.append("|------|----------|------|------|--------|")
        for err in errors:
            test = err["TestName"]
            detail = err["ErrorDetail"][:120].replace("|", "/")
            cat = err["Category"]
            rc = err["RootCause"]
            conf = err["Confidence"]
            L.append(f"| {test} | {detail} | {cat} | {rc} | {conf} |")
        L.append("")

    L.append("---")
    L.append("")
    L.append("*报告由 GitHub Actions 日志分析器错误分析模块生成*")

    report_path = os.path.join(reports_dir, "error_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--since-date", default="")
    parser.add_argument("--until-date", default="")
    args = parser.parse_args()

    base = os.path.abspath(args.base_dir)
    logs_dir = os.path.join(base, "analysis", "logs", args.timestamp)
    reports_dir = os.path.join(base, "analysis", "reports", args.timestamp)

    if not os.path.exists(logs_dir):
        print(f"Logs directory not found: {logs_dir}")
        return
    os.makedirs(reports_dir, exist_ok=True)

    print(f"Analyzing logs in: {logs_dir}")
    all_errors, job_error_map = analyze_all_logs(logs_dir, reports_dir)

    csv_path = generate_error_csv(all_errors, reports_dir)
    print(f"CSV saved: {csv_path} ({len(all_errors)} records)")

    report_path = generate_error_report(all_errors, job_error_map, reports_dir, args.since_date, args.until_date)
    print(f"Report saved: {report_path}")

    print("\n=== Error Analysis Complete ===")


if __name__ == "__main__":
    main()
