"""Curate completed batches08/09 aggregate metrics/configs, never raw artifacts."""
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
    8:[
        '- s1_predw_1e3/1e2/1e1只改变原FSQ Stage1收益监督权重为0.001/0.01/0.1，均从seed42重新训练。',
        '- fsq_ifsq使用tanh(0.8x)限幅尺度；fsq_cs用完整当日截面对投影维度z-score；fsq_rank用当日平均并列rank分8档，反向通过原有界投影STE。不以截面标准化代替等频分档。',
        '- fsq_levels_44442、fsq_levels_binary9、fsq_levels_16162分别使用[4,4,4,4,2]、9个二档维度、[16,16,2]，均为512码；Stage1/Stage2潜在接口仍为128维。',
        '- sinkhorn_quant采用512个可学习128维中心，平方距离/0.1生成logits，20轮可微log-Sinkhorn；当前完整截面train/eval均做软均衡，硬argmax前向、软重建梯度STE，commit=0.25。软质量与硬占用分别记录，N<512时不强求硬码等量。',
        '- bsq_tokenizer为9bit球面二值量化：归一化投影后符号量化至±1/3，再映射回128维；温度0.1，对全部512个球面顶点计算精确联合熵目标，权重0.1、commit权重0。与9维二档FSQ分别保留，不将两者等同。',
    ],
    9:[
        '- fsq_res使用[3,3,3]粗码加LayerNorm残差的[8,8,8]细码，表示求和；有效组合数13824。fsq_group把128维拆成两组64维，各自使用512码，表示拼接；有效组合数262144。分量整数码另存缓存，分组表示在z_q的两半，均不构造联合one-hot。',
        '- fsq_scale学习逐维正尺度exp(log_scale)，log_scale限制[-5,5]，初始为1。fsq_dither仅Stage1训练时在整数网格单位加入U(-0.5,0.5)并截断端点，推理/Stage2恢复硬取整。',
        '- fsq_ortho对原投影维度相关矩阵的非对角项平方取均值，训练权重0.01。fsq_style_orth对FSQ投影与数据中实际个股风格ROC20/STD20/BETA20/VSTD20的相关加权0.01；不使用同日恒定的13个JKP因子充当个股风格。',
        '- fsq_hybrid使用3维离散坐标加4维tanh连续支路，共同投影至128维并同步用于解码和标准Stage2。离散投影按原FSQ初始化，连续支路的联合输出权重初始为零。',
        '- fsq_contrast以同一股票前一完整市场交易日为正样本，同日其他股票为负样本，使用对称InfoNCE，温度0.1、权重0.1。两天都保留完整截面编码；缺失前一市场日时不跨空缺补配。只对训练期样本配对，不读取未来标签。',
    ],
}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch',type=int,choices=[8,9],required=True)
    args=p.parse_args();batch=args.batch;planned=22 if batch==8 else 16;title='核心Stage1与量化器对照' if batch==8 else 'FSQ量化器进一步改造';folder=args.output_dir.resolve();assert folder.is_relative_to(ROOT/'artifacts/research')
    state=json.loads((folder/'progress.json').read_text());assert state['status'] in ('complete','complete_with_failures')
    frame=pd.read_csv(folder/'results.csv')
    for field in ('stage1_training_seconds','stage2_training_seconds','training_seconds','evaluation_backtest_seconds','unsaved_training_seconds_lower_bound'):
        if field not in frame:frame[field]=float('nan')
    models=frame[frame.role.eq('new_model')];complete=models[models.status.eq('complete')]
    assert len(models)==planned
    for row in complete.to_dict('records'):
        assert row['standalone_exact'] and row['stage1_unchanged'] and row['stage1_retrained']
        baseline=frame[(frame.universe==row['universe']) & (frame.variant==row['baseline_variant'])].iloc[0]
        for metric in METRICS:
            assert math.isfinite(row[metric]) and math.isclose(row['delta_'+metric],row[metric]-baseline[metric],abs_tol=1e-12)
    original=json.loads((folder/'original_integrity_after.log').read_text());assert original['source_unchanged']
    lines=[f'# 第{batch:02d}批：{title}','',f"状态：{state['status']}；完整新Stage2结果 {len(complete)}/{planned}；各配置均重新训练Stage1。",'',
        'CSI300/CSI800各Stage2 seed0，Stage1从头训练seed42，Stage2固定seed0并冻结对应新Stage1。训练2009–2020，验证2021–2022，测试2023–2025；验证检查点按原主MSE与原辅助项选择。',
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
    tcols=['universe','variant','status','new_training','reused_reference','training_seconds','stage1_training_seconds','stage2_training_seconds','stage1_completed_epochs','stage1_parameters','unsaved_training_seconds_lower_bound',
        'training_seconds_including_replay_lower_bound','evaluation_backtest_seconds','evaluation_seconds','backtest_seconds',
        'gpu_peak_allocated_mib','gpu_peak_reserved_mib','total_parameters','trainable_parameters','completed_epochs','runtime_workers']
    frame.reindex(columns=tcols).to_csv(folder/'timing_memory.csv',index=False)
    fit=models.training_seconds.sum();extra=models.unsaved_training_seconds_lower_bound.sum();evaluation=models.evaluation_backtest_seconds.sum()
    s1=models.stage1_training_seconds.sum();s2=models.stage2_training_seconds.sum()
    lines+=['',f'**实际资源**。Stage1 fit合计{s1/3600:.2f}小时，Stage2 fit合计{s2/3600:.2f}小时；保存/恢复路径fit合计{fit/3600:.2f}小时，另计中断前未保存训练至少{extra/60:.2f}分钟；'
        f'已记录训练合计至少{(fit+extra)/3600:.2f}小时。预测、独立评估与回测{evaluation/60:.2f}分钟。'
        'fit含训练内验证，不含停机等待。逐项见timing_memory.csv。PyTorch allocated/reserved峰值与整卡占用分别记录；整卡明细仅留本地。','',
        '**实现与比较边界**。','',
        *METHODS[batch],'',
        '**恢复与监控**。每轮原子保存latest检查点及上一份备份，完整保存优化器、调度器、Python/NumPy/Torch/CUDA RNG。'
        '训练进程和独立GPU监控使用用户级系统服务，服务异常自动重启；单阶段最多3次尝试，独立修复最多16次，30分钟内最多3次。'
        '数据加载进程异常时仅降到workers=0；不自动改变模型、样本范围或回测规则。反复失败保留原因并继续其他项目。','',
        '**逐项描述**。仅seed0，不能据此断言稳定性或统计显著。','']
    for row in models.to_dict('records'):
        if row['status']!='complete':lines.append(f"- {row['universe']}/{row['variant']}未完成：{row.get('error','')}");continue
        lines.append(f"- {row['universe']}/{row['variant']}：ΔRankIC {row['delta_RankIC']:+.5f}，"
                     f"Δ净年化 {row['delta_AR']*100:+.3f}个百分点，ΔSharpe {row['delta_Sharpe']:+.3f}，"
                     f"ΔMDD {row['delta_MDD']*100:+.3f}个百分点。")
    lines+=['','Stage1额外相关/时间对比惩罚仅用于训练；验证保留重建损失、对应量化器内在目标及指定权重的原收益MSE。Stage2验证仍使用原主MSE与原辅助项；测试集不参与选择。不同Stage1目标的绝对验证损失不能直接跨方案比较。','',
        '机制定义参考：[BSQ](https://arxiv.org/abs/2406.07548)、[SwAV软均衡](https://github.com/facebookresearch/swav/blob/main/main_swav.py)。','',
        '实现：trace_vq/models/batch89.py、batch89_quantizers.py；相邻日数据trace_vq/data/batch89.py；'
        '训练入口trace_vq/cli/batch89.py，恢复回调trace_vq/training/resilient.py；'
        '串行编排/监控/报告research/run_batch89.py、monitor_batch89.py、report_batch89.py。配置见configs/。','']
    report='\n'.join(lines);(folder/'README.md').write_text(report)
    public=ROOT/'experiments'/folder.name;public.mkdir(exist_ok=True)
    for name in ['results.csv','variant_results.csv','baselines.csv','timing_memory.csv']:
        f=pd.read_csv(folder/name).drop(columns=['output_dir','checkpoint','config','stage1_output_dir','stage1_checkpoint','stage1_config'],errors='ignore')
        (public/name).write_text(portable(f.to_csv(index=False)))
    (public/'README.md').write_text(portable(report));(public/'plan.json').write_text(portable((folder/'plan.json').read_text()))
    for record in state['runs'].values():
        if record.get('role')!='new_model':continue
        for stage,seed,field in ((1,42,'stage1_config'),(2,0,'config')):
            p=public/'configs'/record['universe']/record['variant']/f'stage{stage}_seed{seed}.yaml';p.parent.mkdir(parents=True,exist_ok=True)
            source=Path(record.get('stage1_output_dir',''))/'effective.config.yaml' if stage==1 else Path(record[field])
            if not source.is_file():source=Path(record[field])
            p.write_text(portable(source.read_text()))
    validation=json.loads((folder/'validation.json').read_text());validation.update(standalone_exact_runs=len(complete),original_unchanged=True,
        source_sha256=state['source_sha256'],recoveries=len(state['recoveries']),unsaved_fit_seconds_lower_bound=float(extra))
    (public/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    path=ROOT/'experiments/index.csv';index=pd.read_csv(path)
    if folder.name not in set(index.batch):
        row=dict(batch=folder.name,description=f'第{batch:02d}批：{title}',universes='CSI300 / CSI800',seeds='Stage1 42; Stage2 0',
            records=f'{len(complete)}/{planned}组两阶段完整结果，2份复用FSQ基线',finding='仅seed0；全部指标、对应FSQ差值及资源记录见报告',readme=folder.name+'/README.md')
        pd.concat([index,pd.DataFrame([row])],ignore_index=True).to_csv(path,index=False)
    path=ROOT/'experiments/README.md';text=path.read_text()
    if f']({folder.name}/README.md)' not in text:path.write_text(text+f'\n第{batch:02d}批结果见 [{folder.name}]({folder.name}/README.md)。\n')
    subprocess.run([sys.executable,'research/organize_experiment_results.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'research/audit_git_scope.py'],cwd=ROOT,check=True)
    result=dict(status='passed',new_training_complete=len(complete),planned_new_training=planned,public_summary=str(public),batch_status=state['status'])
    (folder/'report_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
