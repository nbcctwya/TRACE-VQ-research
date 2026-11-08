"""Curate completed batches06/07 aggregate metrics/configs, never raw artifacts."""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.report_batch34 import portable,number,METRICS


METHODS={
    6:[
        '- fsq_soft将归一化取整坐标和连续坐标拼成6维并投影至128维；初始化仅保留原离散映射，训练后可利用连续信息。CA继续基于原冻结重建误差。',
        '- fsq_level_emb采用三组8×128档位表相加，初始化为原from_grid的可加映射；fsq_cross在其上增加三组64×128两两组合表，零初始化。两者引入的参数量不同，保留实际计数。',
        '- fsq_grid_smooth对当前码的有效六向邻格潜在因子值加均方差拉普拉斯惩罚，权重0.01；边两端使用完全相同CA向量。惩罚仅用于训练，不加入验证检查点目标。',
        '- fsq_shrink使用512个子码截距和64个粗父码截距，每维两档合并。惩罚权重0.01，逐码系数20/(训练期频数+20)，未见过的码使用父截距；频数只从训练期计算。',
        '- fsq_code_aug仅训练时，对距半整数边界不超过0.1整数单位的坐标以0.1概率翻到另一邻档；重算完整码、重建向量和CA误差。验证/测试无随机扰动。',
        '- kan_head只替换潜在因子值头：归一化3维取整坐标输入[3,16,128]的边函数样条网络，5段均匀网格、3阶B样条加SiLU基函数，隐藏层tanh。载荷保留原FSQ+CA；可画响应曲线不自动代表经济解释。',
    ],
    7:[
        '- code_hypergraph在Stage2以512个格点为超边中心，同码或曼哈顿距离1的股票参与同一超边；按超边/顶点度归一化做一次同日完整截面消息传递，残差输出零初始化，无标签输入。',
        '- hyper_loading以16维code embedding生成rank4载荷增量的输出矩阵，输入低秩投影共享；分别加到先验/潜在载荷，按更新后载荷计算原范数辅助项。不生成整个大网络。',
        '- film_code取消原前置结构token，改用学习CLS；由16维code embedding为每个时间块生成gamma/beta，在块输入做(1+gamma)*x+beta，零初始化。FSQ仍用于原下游融合和因子头。',
        '- hopfield_retrieval复用第三批坐标/交易日缓存和原冻结h；每码最多保留8项，从同码或六向邻码选择最近16项。32维归一化键/查询、逆温度4、一次softmax关联更新；值为历史h与已到期的处理后主标签，残差输出零初始化。',
        '- 标签成熟规则为source_day+5严格早于query_day，最小间隔6个完整市场交易日，排除自身和同日样本。按日期先解封过去标签，再生成候选；允许验证/测试期间已知的历史结果进入记忆，不用未到期标签，也不对验证/测试标签做梯度更新。',
        '- 候选按固定日期/索引和码选择，不按标签大小选择；空记忆输出为零。使用未来标签扰动、完整交易日间隔和严格恢复测试；检索是受限候选池原型，不代表无限记忆或论文完整应用效果。',
    ],
}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch',type=int,choices=[6,7],required=True)
    args=p.parse_args();batch=args.batch;planned=14 if batch==6 else 8;title='FSQ网格表达与预测头' if batch==6 else '码条件化与关系检索';folder=args.output_dir.resolve();assert folder.is_relative_to(ROOT/'artifacts/research')
    state=json.loads((folder/'progress.json').read_text());assert state['status'] in ('complete','complete_with_failures')
    frame=pd.read_csv(folder/'results.csv');models=frame[frame.role.eq('new_model')];complete=models[models.status.eq('complete')]
    assert len(models)==planned
    for row in complete.to_dict('records'):
        assert row['standalone_exact'] and row['stage1_unchanged'] and not row['stage1_retrained']
        baseline=frame[(frame.universe==row['universe']) & (frame.variant==row['baseline_variant'])].iloc[0]
        for metric in METRICS:
            assert math.isfinite(row[metric]) and math.isclose(row['delta_'+metric],row[metric]-baseline[metric],abs_tol=1e-12)
    original=json.loads((folder/'original_integrity_after.log').read_text());assert original['source_unchanged']
    lines=[f'# 第{batch:02d}批：{title}','',f"状态：{state['status']}；完整新Stage2结果 {len(complete)}/{planned}；无新增Stage1训练。",'',
        'CSI300/CSI800各Stage2 seed0，Stage1复用对应seed42。训练2009–2020，验证2021–2022，测试2023–2025；验证检查点按原主MSE与原辅助项选择。',
        'TopkDropout沿用all报价、top30/drop5、risk0.95、close、买入0.0005/卖出0.0015、最低费0。',
        '本批全部以原FSQ为对照，保持128个潜在因子；以下AR/MDD/换手为百分比，CSV保留比例。','',
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
        *METHODS[batch],'',
        '**恢复与监控**。每轮原子保存latest检查点及上一份备份，完整保存优化器、调度器、Python/NumPy/Torch/CUDA RNG。'
        '训练进程和独立GPU监控使用用户级系统服务，服务异常自动重启；单阶段最多3次尝试，独立修复最多8次，30分钟内最多3次。'
        '数据加载进程异常时仅降到workers=0；不自动改变模型、样本范围或回测规则。反复失败保留原因并继续其他项目。','',
        '**逐项描述**。仅seed0，不能据此断言稳定性或统计显著。','']
    for row in models.to_dict('records'):
        if row['status']!='complete':lines.append(f"- {row['universe']}/{row['variant']}未完成：{row.get('error','')}");continue
        lines.append(f"- {row['universe']}/{row['variant']}：ΔRankIC {row['delta_RankIC']:+.5f}，"
                     f"Δ净年化 {row['delta_AR']*100:+.3f}个百分点，ΔSharpe {row['delta_Sharpe']:+.3f}，"
                     f"ΔMDD {row['delta_MDD']*100:+.3f}个百分点。")
    lines+=['','机制定义参考：[KAN](https://arxiv.org/abs/2404.19756)、[现代Hopfield检索](https://arxiv.org/abs/2008.02217)。','',
        '实现：trace_vq/models/batch67.py；因果检索缓存trace_vq/data/hopfield.py；'
        '训练入口trace_vq/cli/batch67.py，恢复回调trace_vq/training/resilient.py；'
        '串行编排/监控/报告research/run_batch67.py、monitor_batch67.py、report_batch67.py。配置见configs/。','']
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
        row=dict(batch=folder.name,description=f'第{batch:02d}批：{title}',universes='CSI300 / CSI800',seeds='Stage1 42; Stage2 0',
            records=f'{len(complete)}/{planned}次新训练完整结果，2份复用FSQ基线',finding='仅seed0；全部指标、对应FSQ差值及资源记录见报告',readme=folder.name+'/README.md')
        pd.concat([index,pd.DataFrame([row])],ignore_index=True).to_csv(path,index=False)
    path=ROOT/'experiments/README.md';text=path.read_text()
    if f']({folder.name}/README.md)' not in text:path.write_text(text+f'\n第{batch:02d}批结果见 [{folder.name}]({folder.name}/README.md)。\n')
    subprocess.run([sys.executable,'research/organize_experiment_results.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'research/audit_git_scope.py'],cwd=ROOT,check=True)
    result=dict(status='passed',new_training_complete=len(complete),planned_new_training=planned,public_summary=str(public),batch_status=state['status'])
    (folder/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
