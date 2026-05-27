#!/usr/bin/env python3
"""GitHub Actions Log Analyzer - Report Generator"""
import csv, os, argparse
from datetime import datetime
from collections import defaultdict

def pct(n, d): return round(n/d*100,1) if d>0 else 0
def safe(t, m=40):
    t = t.replace("|","/")
    return t[:m]+"..." if len(t)>m else t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--repo", default="sgl-project/sgl-kernel-npu")
    args = parser.parse_args()
    base = os.path.abspath(args.base_dir)
    rdir = os.path.join(base, "analysis", "reports", args.timestamp)
    odir = os.path.join(base, "analysis", "optimization", args.timestamp)
    os.makedirs(odir, exist_ok=True)
    with open(os.path.join(rdir,"all_runs.csv"), encoding="utf-8-sig") as f: runs=list(csv.DictReader(f))
    with open(os.path.join(rdir,"all_jobs.csv"), encoding="utf-8-sig") as f: jobs=list(csv.DictReader(f))
    dt=datetime.now().strftime("%Y-%m-%d %H:%M:%S"); repo=args.repo
    wr_d=defaultdict(list); wj_d=defaultdict(list)
    for r in runs: wr_d[r["WorkflowName"]].append(r)
    for j in jobs: wj_d[j["WorkflowName"]].append(j)

    L=[]
    L.append(f"# {repo} CI/CD Pipeline Analysis Report")
    L.append(""); L.append(f"**Generated**: {dt}"); L.append(f"**Repository**: {repo}")
    L.append("**Scope**: All completed runs (success + failure) across 3 workflows")
    L.append(f"**Data**: {len(runs)} runs, {len(jobs)} jobs"); L.append(""); L.append("---"); L.append("")

    L.append("## 1. Overview"); L.append("")
    for wf in ["build_and_release","daily-build-test","pr-test-npu"]:
        wr=wr_d.get(wf,[]); wj=wj_d.get(wf,[])
        if not wr: continue
        rs=sum(1 for r in wr if r["Conclusion"]=="success"); rf=len(wr)-rs
        js=sum(1 for j in wj if j["JobConclusion"]=="success"); jf=sum(1 for j in wj if j["JobConclusion"]=="failure")
        jc=sum(1 for j in wj if j["JobConclusion"]=="cancelled"); jsk=sum(1 for j in wj if j["JobConclusion"]=="skipped")
        eff=len(wj)-jc-jsk
        L.append(f"### {wf}"); L.append("")
        L.append("| Metric | Value |"); L.append("|--------|-------|")
        L.append(f"| Total Runs | {len(wr)} |"); L.append(f"| Run Success / Failure | {rs} / {rf} |")
        L.append(f"| Run Success Rate | {pct(rs,len(wr))}% |"); L.append(f"| Total Jobs | {len(wj)} |")
        L.append(f"| Job Success / Failure / Cancelled / Skipped | {js} / {jf} / {jc} / {jsk} |")
        L.append(f"| Job Pass Rate (excl cancelled/skipped) | {pct(js,eff)}% |"); L.append("")

    trs=sum(1 for r in runs if r["Conclusion"]=="success")
    tjs=sum(1 for j in jobs if j["JobConclusion"]=="success"); tjf=sum(1 for j in jobs if j["JobConclusion"]=="failure")
    tjc=sum(1 for j in jobs if j["JobConclusion"]=="cancelled"); tjsk=sum(1 for j in jobs if j["JobConclusion"]=="skipped")
    teff=len(jobs)-tjc-tjsk
    L.append("### Grand Total"); L.append("")
    L.append("| Metric | Value |"); L.append("|--------|-------|")
    L.append(f"| Total Runs | {len(runs)} |"); L.append(f"| Run Success / Failure | {trs} / {len(runs)-trs} |")
    L.append(f"| Run Success Rate | {pct(trs,len(runs))}% |"); L.append(f"| Total Jobs | {len(jobs)} |")
    L.append(f"| Job Success / Failure / Cancelled / Skipped | {tjs} / {tjf} / {tjc} / {tjsk} |")
    L.append(f"| Job Pass Rate (excl cancelled/skipped) | {pct(tjs,teff)}% |"); L.append("")
    L.append("---"); L.append("")

    L.append("## 2. Per-Run Job Statistics (Recent 20 per workflow)"); L.append("")
    for wf in ["build_and_release","daily-build-test","pr-test-npu"]:
        wr=wr_d.get(wf,[])
        if not wr: continue
        L.append(f"### {wf}"); L.append("")
        L.append("| Run | Title | Conclusion | Total | Success | Failure | Cancelled | Pass Rate |")
        L.append("|-----|-------|------------|-------|---------|---------|-----------|-----------|")
        for r in sorted(wr, key=lambda r:r["CreatedAt"],reverse=True)[:20]:
            rj=[j for j in wj_d.get(wf,[]) if j["RunId"]==r["RunId"]]
            s=sum(1 for j in rj if j["JobConclusion"]=="success"); f=sum(1 for j in rj if j["JobConclusion"]=="failure")
            c=sum(1 for j in rj if j["JobConclusion"]=="cancelled"); eff=len(rj)-c; rate=pct(s,eff)
            mark="OK" if r["Conclusion"]=="success" else "FAIL"; title=safe(r.get("Title","")); url=r.get("HtmlUrl","")
            L.append(f"| [{r['RunId']}]({url}) | {title} | {mark} | {len(rj)} | {s} | {f} | {c} | {rate}% |")
        L.append("")
    L.append("---"); L.append("")

    L.append("## 3. Stability Analysis"); L.append("")
    for wf in ["pr-test-npu","daily-build-test"]:
        wj=[j for j in wj_d.get(wf,[]) if j["JobConclusion"]!="cancelled"]
        if not wj: continue
        L.append(f"### {wf} Job Stability"); L.append("")
        L.append("| Job Name | Total | Success | Failure | Pass Rate |"); L.append("|----------|-------|---------|---------|-----------|")
        js={}
        for j in wj:
            n=j["JobName"]
            if n not in js: js[n]={"total":0,"success":0,"failure":0}
            js[n]["total"]+=1
            if j["JobConclusion"]=="success": js[n]["success"]+=1
            elif j["JobConclusion"]=="failure": js[n]["failure"]+=1
        for n in sorted(js, key=lambda x:js[x]["total"],reverse=True):
            st=js[n]; L.append(f"| {n} | {st['total']} | {st['success']} | {st['failure']} | {pct(st['success'],st['total'])}% |")
        L.append("")

    L.append("### PR Workflow E2E Duration (Target: <= 60 min)"); L.append("")
    L.append("| Run | Title | Conclusion | Max Core Job Duration | Meets Target |")
    L.append("|-----|-------|------------|----------------------|-------------|")
    for r in sorted(wr_d.get("pr-test-npu",[]), key=lambda r:r["CreatedAt"],reverse=True)[:30]:
        rj=[j for j in wj_d.get("pr-test-npu",[]) if j["RunId"]==r["RunId"] and j["JobConclusion"]!="cancelled" and "Check changed files" not in j["JobName"] and j["JobName"]!="finish"]
        if not rj: continue
        durs=[float(j["DurationMin"]) for j in rj if float(j.get("DurationMin",0))>0]
        mx=round(max(durs),1) if durs else 0; meets="YES" if mx<=60 else "NO"
        title=safe(r.get("Title",""),30); url=r.get("HtmlUrl","")
        L.append(f"| [{r['RunId']}]({url}) | {title} | {r['Conclusion']} | {mx} min | {meets} |")
    L.append(""); L.append("---"); L.append("")

    L.append("## 4. Execution Time Analysis"); L.append("")
    tj=[j for j in jobs if j["JobConclusion"]!="cancelled" and float(j.get("DurationMin",0))>0]
    tj.sort(key=lambda j:float(j["DurationMin"]),reverse=True)
    L.append("### 4.1 Top 30 Longest Jobs (All Workflows)"); L.append("")
    L.append("| Rank | Workflow | Job Name | Run ID | Duration (min) | Conclusion |")
    L.append("|------|----------|----------|--------|---------------|------------|")
    for i,j in enumerate(tj[:30],1):
        L.append(f"| {i} | {j['WorkflowName']} | {j['JobName']} | {j['RunId']} | {float(j['DurationMin'])} | {j['JobConclusion']} |")
    L.append("")

    L.append("### 4.2 Average Duration by Job Name"); L.append("")
    L.append("| Workflow | Job Name | Count | Avg (min) | Max (min) | Min (min) |")
    L.append("|----------|----------|-------|-----------|-----------|-----------|")
    ds={}
    for j in tj:
        k=(j["WorkflowName"],j["JobName"])
        if k not in ds: ds[k]=[]
        ds[k].append(float(j["DurationMin"]))
    for (wf,nm),durs in sorted(ds.items(), key=lambda x:-max(x[1])):
        L.append(f"| {wf} | {nm} | {len(durs)} | {round(sum(durs)/len(durs),1)} | {round(max(durs),1)} | {round(min(durs),1)} |")
    L.append(""); L.append("---"); L.append("")

    L.append("## 5. Failure Analysis"); L.append("")
    fj=[j for j in jobs if j["JobConclusion"]=="failure"]
    L.append("### 5.1 Failed Jobs by Workflow"); L.append("")
    L.append("| Workflow | Failed Jobs | Total Effective Jobs | Failure Rate |"); L.append("|----------|------------|--------------------|-------------|")
    for wf in ["build_and_release","daily-build-test","pr-test-npu"]:
        wf_f=[j for j in fj if j["WorkflowName"]==wf]; wf_e=[j for j in wj_d.get(wf,[]) if j["JobConclusion"]!="cancelled"]
        L.append(f"| {wf} | {len(wf_f)} | {len(wf_e)} | {pct(len(wf_f),len(wf_e))}% |")
    L.append("")

    L.append("### 5.2 Failed Jobs by Job Name"); L.append("")
    L.append("| Workflow | Job Name | Fail Count | Common Failed Steps |"); L.append("|----------|----------|------------|---------------------|")
    fbn={}
    for j in fj:
        k=(j["WorkflowName"],j["JobName"])
        if k not in fbn: fbn[k]={"count":0,"steps":set()}
        fbn[k]["count"]+=1
        for s in j.get("FailedStepsNames","").split("; "):
            if s: fbn[k]["steps"].add(s)
    for (wf,nm),info in sorted(fbn.items(), key=lambda x:-x[1]["count"]):
        ss=", ".join(sorted(info["steps"]))[:80]
        L.append(f"| {wf} | {nm} | {info['count']} | {ss} |")
    L.append(""); L.append("---"); L.append("")
    L.append(f"*Report generated by GitHub Actions Log Analyzer v2 | Data source: {repo}*")

    with open(os.path.join(rdir,"full_analysis_report.md"),"w",encoding="utf-8") as f: f.write("\n".join(L))
    print(f"Report saved: {os.path.join(rdir,'full_analysis_report.md')}")

    # Optimization report
    O=[]
    O.append(f"# {repo} CI/CD Pipeline Optimization Proposal"); O.append("")
    O.append(f"Generated: {dt}"); O.append("Scope: Reduce total pipeline execution time without modifying test scripts"); O.append("")
    pt=[j for j in wj_d.get("pr-test-npu",[]) if j["JobConclusion"]!="cancelled" and "Check changed files" not in j["JobName"] and j["JobName"]!="finish" and float(j.get("DurationMin",0))>0]
    O.append("## 1. Current PR Workflow Time Breakdown"); O.append("")
    O.append("| Job Name | Count | Avg (min) | Max (min) |"); O.append("|----------|-------|-----------|-----------|")
    pd={}
    for j in pt:
        n=j["JobName"]
        if n not in pd: pd[n]=[]
        pd[n].append(float(j["DurationMin"]))
    for n in sorted(pd, key=lambda n:-max(pd[n])):
        durs=pd[n]; O.append(f"| {n} | {len(durs)} | {round(sum(durs)/len(durs),1)} | {round(max(durs),1)} |")
    O.append("")
    O.append("## 2. Bottleneck Analysis"); O.append("")
    for n in sorted(pd, key=lambda n:-sum(pd[n])/len(pd[n])):
        durs=pd[n]; O.append(f"- **{n}**: avg {round(sum(durs)/len(durs),1)} min, max {round(max(durs),1)} min")
    O.append("")
    O.append("## 3. Optimization Proposals"); O.append("")
    O.append("### Proposal 1: Split test-all-build into Parallel Sub-Jobs"); O.append("")
    O.append("Current: test-all-build runs all DeepEP tests sequentially (45+ steps, ~63 min)")
    O.append("Proposal: Split into 3 parallel jobs by test type:"); O.append("")
    O.append("| New Job | Tests Included | Est. Duration |"); O.append("|---------|---------------|--------------|")
    O.append("| test-intranode | test_intranode (13 variants) | ~15 min |")
    O.append("| test-low-latency-moe | test_low_latency (8) + test_fused_deep_moe (8) + test_mixed_running (7) | ~25 min |")
    O.append("| test-combine-misc | test_combine (1) + test_generalization_fused_deep_moe (1) | ~10 min |"); O.append("")
    O.append("Expected improvement: Wall-clock time from ~63 min to ~25 min (60% reduction)"); O.append("")
    O.append("### Proposal 2: Share Build Artifact Across Jobs"); O.append("")
    O.append("Current: test-all-build, test-build-deepep-a3, test-build-deepep-a2 each build DeepEP independently")
    O.append("Proposal: Create a dedicated build job, upload wheel as GitHub Actions artifact")
    O.append("Expected improvement: Save 20-30 min of redundant build time per run"); O.append("")
    O.append("### Proposal 3: Conditional Internode Testing"); O.append("")
    O.append("Current: test_internode_a2 runs on every PR, fails ~90% of the time, blocks workflow for 3 hours")
    O.append("Proposal: continue-on-error: true / schedule only / reduce timeout from 10800s to 1800s")
    O.append("Expected improvement: PR workflow from 3+ hours to <30 min when K8s is unavailable"); O.append("")
    O.append("### Proposal 4: Optimize enumerate_test Shell Scripts"); O.append("")
    O.append("Current: enumerate scripts run tests sequentially for each parameter combination")
    O.append("Proposal: Run parameter combinations in parallel using background processes or GNU parallel")
    O.append("Expected improvement: 30-50% reduction in daily-build-test job duration"); O.append("")
    O.append("### Proposal 5: Add sgl_kernel_npu Operator Tests to CI"); O.append("")
    O.append("Current: 36 operator tests in tests/python/sgl_kernel_npu/ are not in any CI workflow")
    O.append("Proposal: Add a new job in daily-build-test.yml or pr-test-npu.yml"); O.append("")
    O.append("## 4. Summary of Expected Improvements"); O.append("")
    O.append("| Proposal | Target | Expected Time Saving | Priority |"); O.append("|----------|--------|---------------------|----------|")
    O.append("| Split test-all-build | PR workflow | ~38 min (63 to 25 min) | P0 |")
    O.append("| Share build artifact | PR workflow | ~20 to 30 min | P0 |")
    O.append("| Conditional internode | PR workflow | ~2.5 hours (when K8s down) | P0 |")
    O.append("| Parallel enumerate | Daily workflow | 30 to 50 pct job time | P1 |")
    O.append("| Add operator tests | Coverage | N/A (new tests) | P1 |"); O.append("")
    with open(os.path.join(odir,"optimization_proposal.md"),"w",encoding="utf-8") as f: f.write("\n".join(O))
    print(f"Optimization report saved: {os.path.join(odir,'optimization_proposal.md')}")

    # Excel
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        wb=Workbook()
        hf=Font(bold=True,color="FFFFFF",size=11); hfl=PatternFill(start_color="4472C4",end_color="4472C4",fill_type="solid")
        tb=Border(left=Side(style='thin'),right=Side(style='thin'),top=Side(style='thin'),bottom=Side(style='thin'))
        def sh(ws,mc):
            for c in range(1,mc+1):
                cl=ws.cell(row=1,column=c);cl.font=hf;cl.fill=hfl;cl.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True);cl.border=tb
        def aw(ws):
            for col in ws.columns:
                ml=0;cn=col[0].column_letter
                for cell in col:
                    try:
                        if len(str(cell.value))>ml:ml=len(str(cell.value))
                    except:pass
                ws.column_dimensions[cn].width=min(ml+4,60)

        ws1=wb.active;ws1.title="Per-Run Stats"
        h1=["Run ID","URL","Workflow","Title","Conclusion","Total Jobs","Success","Failure","Cancelled","Pass Rate"]
        for ci,h in enumerate(h1,1):ws1.cell(row=1,column=ci,value=h)
        sh(ws1,len(h1));ri=2
        for r in sorted(runs,key=lambda x:x["CreatedAt"],reverse=True):
            rj=[j for j in jobs if j["RunId"]==r["RunId"]]
            s=sum(1 for j in rj if j["JobConclusion"]=="success");f=sum(1 for j in rj if j["JobConclusion"]=="failure")
            c=sum(1 for j in rj if j["JobConclusion"]=="cancelled");eff=len(rj)-c;rate=f"{pct(s,eff)}%"
            vals=[r["RunId"],r.get("HtmlUrl",""),r["WorkflowName"],safe(r.get("Title",""),50),r["Conclusion"],len(rj),s,f,c,rate]
            for ci,v in enumerate(vals,1):cl=ws1.cell(row=ri,column=ci,value=v);cl.border=tb;cl.alignment=Alignment(vertical='center',wrap_text=True)
            ri+=1
        aw(ws1)

        ws2=wb.create_sheet("Job Duration")
        h2=["Workflow","Job Name","Count","Avg (min)","Max (min)","Min (min)","Success","Failure","Pass Rate"]
        for ci,h in enumerate(h2,1):ws2.cell(row=1,column=ci,value=h)
        sh(ws2,len(h2));ri=2
        for (wf,nm),durs in sorted(ds.items(),key=lambda x:-max(x[1])):
            avg=round(sum(durs)/len(durs),1);mx=round(max(durs),1);mn=round(min(durs),1)
            wf_j=[j for j in wj_d.get(wf,[]) if j["JobName"]==nm and j["JobConclusion"]!="cancelled"]
            s=sum(1 for j in wf_j if j["JobConclusion"]=="success");f=sum(1 for j in wf_j if j["JobConclusion"]=="failure")
            rate=f"{pct(s,len(wf_j))}%"
            vals=[wf,nm,len(durs),avg,mx,mn,s,f,rate]
            for ci,v in enumerate(vals,1):cl=ws2.cell(row=ri,column=ci,value=v);cl.border=tb;cl.alignment=Alignment(vertical='center',wrap_text=True)
            ri+=1
        aw(ws2)

        ws3=wb.create_sheet("Stability Summary")
        h3=["Workflow","Job Name","Total Runs","Success","Failure","Pass Rate"]
        for ci,h in enumerate(h3,1):ws3.cell(row=1,column=ci,value=h)
        sh(ws3,len(h3));ri=2
        for wf in ["pr-test-npu","daily-build-test","build_and_release"]:
            wj=[j for j in wj_d.get(wf,[]) if j["JobConclusion"]!="cancelled"];jss={}
            for j in wj:
                n=j["JobName"]
                if n not in jss:jss[n]={"total":0,"success":0,"failure":0}
                jss[n]["total"]+=1
                if j["JobConclusion"]=="success":jss[n]["success"]+=1
                elif j["JobConclusion"]=="failure":jss[n]["failure"]+=1
            for n in sorted(jss,key=lambda x:-jss[x]["total"]):
                st=jss[n];rate=f"{pct(st['success'],st['total'])}%"
                vals=[wf,n,st["total"],st["success"],st["failure"],rate]
                for ci,v in enumerate(vals,1):cl=ws3.cell(row=ri,column=ci,value=v);cl.border=tb;cl.alignment=Alignment(vertical='center',wrap_text=True)
                ri+=1
        aw(ws3)

        xp=os.path.join(rdir,"analysis_report.xlsx");wb.save(xp);print(f"Excel saved: {xp}")
    except ImportError:print("Warning: openpyxl not installed, skipping Excel")
    print("\n=== Report Generation Complete ===")

if __name__=="__main__":main()
