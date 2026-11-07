"""Curate completed batch05 aggregate metrics/configs, never raw artifacts."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.report_batch34 import portable,number,METRICS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();folder=args.output_dir.resolve();assert folder.is_relative_to(ROOT/'artifacts/research')
    state=json.loads((folder/'progress.json').read_text());assert state['status'] in ('complete','complete_with_failures')
    frame=pd.read_csv(folder/'results.csv');models=frame[frame.role.eq('new_model')];complete=models[models.status.eq('complete')]
    assert len(models)==8
    for row in complete.to_dict('records'):
        assert row['standalone_exact'] and row['stage1_unchanged'] and not row['stage1_retrained']
        baseline=frame[(frame.universe==row['universe']) & (frame.variant==row['baseline_variant'])].iloc[0]
        for metric in METRICS:
            assert math.isfinite(row[metric]) and math.isclose(row['delta_'+metric],row[metric]-baseline[metric],abs_tol=1e-12)
    original=json.loads((folder/'original_integrity_after.log').read_text());assert original['source_unchanged']
    lines=['# 第05批：新时间模块与截面交互','',f"状态：{state['status']}；完整新Stage2结果 {len(complete)}/8；无新增Stage1训练。",'',
        'CSI300/CSI800各Stage2 seed0，Stage1复用对应seed42。训练2009–2020，验证2021–2022，测试2023–2025；验证检查点按原主MSE与原辅助项选择。',
        'TopkDropout沿用all报价、top30/drop5、risk0.95、close、买入0.0005/卖出0.0015、最低费0。',
        'raven_context_experts对照FSQ，其余对照VQ。以下AR/MDD/换手为百分比，CSV保留比例。','',
        '| 股票池 | 方案 | 对照 | IC | RankIC | 净年化 % | Sharpe | 最大回撤 % | 日均换手 % |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in frame.to_dict('records'):
        control=row.get('baseline_variant');control='' if pd.isna(control) else control
        lines.append(f"| {row['universe']} | {row['variant']} | {control} | "+' | '.join(number(row.get(m),m) for m in METRICS)+' |')
    lines+=['','**相对对应基线的差值**。AR/MDD/换手为百分点，正MDD差值表示回撤减少。','',
        '| 股票池 | 方案 | ΔIC | ΔRankIC | Δ年化 | ΔSharpe | Δ回撤 | Δ换手 |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in models.to_dict('records'):
        lines.append(f"| {row['universe']} | {row['variant']} | "+' | '.join(number(row.get('delta_'+m),m) for m in METRICS)+' |')
    tcols=['universe','variant','status','new_training','reused_reference','training_seconds','unsaved_training_seconds_lower_bound',
        'training_seconds_including_replay_lower_bound','evaluation_backtest_seconds','evaluation_seconds','backtest_seconds',
        'gpu_peak_allocated_mib','gpu_peak_reserved_mib','total_parameters','trainable_parameters','completed_epochs','runtime_workers']
    frame.reindex(columns=tcols).to_csv(folder/'timing_memory.csv',index=False)
    fit=complete.training_seconds.sum();extra=complete.unsaved_training_seconds_lower_bound.sum();evaluation=complete.evaluation_backtest_seconds.sum()
    lines+=['',f'**实际资源**。保存/恢复路径fit合计{fit/3600:.2f}小时，另计中断前未保存训练至少{extra/60:.2f}分钟；'
        f'已记录训练合计至少{(fit+extra)/3600:.2f}小时。预测、独立评估与回测{evaluation/60:.2f}分钟。'
        'fit含训练内验证，不含停机等待。逐项见timing_memory.csv。PyTorch allocated/reserved峰值与整卡占用分别记录；整卡明细仅留本地。','',
        '**实现与比较边界**。','',
        '- KDA使用已有PyTorch的参考递推，逐key通道遗忘门、delta更新、q/k L2归一化、4阶因果短卷积及sigmoid输出门。环境没有FLA，未安装或升级包；不据此宣称FLA融合内核性能或完整复现Kimi Linear大模型。保留Stage2前后的MLP/归一化，结构读取token改放最后且不使用RoPE，以便因果递推读取全部历史。',
        '- mHC使用2条残差流，动态正pre/post映射和20轮log-space Sinkhorn残差映射，分别作用于原注意力/FFN两条残差连接；输出取流均值。保留原Stage2 post-norm约定，未采用论文的大模型混合精度融合内核。',
        '- 多窗口专家使用FSQ路由，分别读取5/10/20日；5/10日塔先裁剪原始输入再做DLinear，采用学习CLS而不注入完整20日FSQ信息。共享分支保留完整20日输入。路由专家数由2变3，新增两个时间塔；所有时间塔均计算，top1不等于跳过其他时间塔计算。需同时考察容量和耗时差异。',
        '- 跨股票注意力在同一交易日的完整股票截面上、原融合/载荷头之前计算；输出投影零初始化。没有跨日期读取，也不读取标签。','',
        '**恢复与监控**。每轮原子保存latest检查点及上一份备份，完整保存优化器、调度器、Python/NumPy/Torch/CUDA RNG。'
        '训练进程和独立GPU监控使用用户级系统服务，服务异常自动重启；单阶段最多3次尝试，独立修复最多8次，30分钟内最多3次。'
        '数据加载进程异常时仅降到workers=0；不自动改变模型、样本范围或回测规则。反复失败保留原因并继续其他项目。','',
        '**逐项描述**。仅seed0，不能据此断言稳定性或统计显著。','']
    for row in models.to_dict('records'):
        if row['status']!='complete':lines.append(f"- {row['universe']}/{row['variant']}未完成：{row.get('error','')}");continue
        lines.append(f"- {row['universe']}/{row['variant']}：ΔRankIC {row['delta_RankIC']:+.5f}，"
                     f"Δ净年化 {row['delta_AR']*100:+.3f}个百分点，ΔSharpe {row['delta_Sharpe']:+.3f}，"
                     f"ΔMDD {row['delta_MDD']*100:+.3f}个百分点。")
    lines+=['','机制核对来源：[Kimi Linear](https://arxiv.org/abs/2510.26692)、'
        '[FLA KDA层](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py)、'
        '[FLA参考算子](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py)、'
        '[mHC](https://arxiv.org/abs/2512.24880)。','',
        '实现：trace_vq/models/batch05.py、trace_vq/models/prediction/batch05_temporal.py；'
        '训练入口trace_vq/cli/batch05.py，恢复回调trace_vq/training/resilient.py；'
        '编排/监控/报告research/run_batch05.py、monitor_batch05.py、report_batch05.py。配置见configs/。','']
    report='\n'.join(lines);(folder/'README.md').write_text(report)
    public=ROOT/'experiments'/folder.name;public.mkdir(exist_ok=True)
    for name in ['results.csv','variant_results.csv','baselines.csv','timing_memory.csv']:
        f=pd.read_csv(folder/name).drop(columns=['output_dir','checkpoint','config'],errors='ignore')
        (public/name).write_text(portable(f.to_csv(index=False)))
    (public/'README.md').write_text(portable(report));(public/'plan.json').write_text(portable((folder/'plan.json').read_text()))
    for record in state['runs'].values():
        if record.get('role')!='new_model':continue
        p=public/'configs'/record['universe']/record['variant']/'seed0.yaml';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(portable(Path(record['config']).read_text()))
    validation=json.loads((folder/'validation.json').read_text());validation.update(standalone_exact_runs=len(complete),original_unchanged=True,
        source_sha256=state['source_sha256'],recoveries=len(state['recoveries']),unsaved_fit_seconds_lower_bound=float(extra))
    (public/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    path=ROOT/'experiments/index.csv';index=pd.read_csv(path)
    if folder.name not in set(index.batch):
        row=dict(batch=folder.name,description='第05批：新时间模块与截面交互',universes='CSI300 / CSI800',seeds='Stage1 42; Stage2 0',
            records=f'{len(complete)}/8次新训练完整结果，4份复用基线',finding='仅seed0；KDA/mHC/多窗口/截面注意力，全部指标及资源差异见报告',readme=folder.name+'/README.md')
        pd.concat([index,pd.DataFrame([row])],ignore_index=True).to_csv(path,index=False)
    path=ROOT/'experiments/README.md';text=path.read_text()
    if f']({folder.name}/README.md)' not in text:path.write_text(text+f'\n第05批结果见 [{folder.name}]({folder.name}/README.md)。\n')
    subprocess.run([sys.executable,'research/organize_experiment_results.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'research/audit_git_scope.py'],cwd=ROOT,check=True)
    result=dict(status='passed',new_training_complete=len(complete),planned_new_training=8,public_summary=str(public),batch_status=state['status'])
    (folder/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
