"""
可视化工具和报告生成系统
提供benchmark和profiling结果的可视化分析功能
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Optional imports with fallbacks
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class BenchmarkVisualizer:
    """基准测试结果可视化器"""
    
    def __init__(self, output_dir: Path):
        """初始化可视化器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置matplotlib样式（如果可用）
        if MATPLOTLIB_AVAILABLE:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 设置seaborn样式
            sns.set_style("whitegrid")
            sns.set_palette("husl")
    
    def visualize_performance_comparison(self, results: List[Dict], save_path: Optional[Path] = None) -> Path:
        """可视化性能对比"""
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available, skipping performance comparison visualization")
            # Create a simple text file with results
            if save_path is None:
                save_path = self.output_dir / "performance_comparison.txt"
            
            with open(save_path, 'w') as f:
                f.write("Performance Comparison Results\n")
                f.write("=" * 40 + "\n")
                for result in results:
                    f.write(f"Method: {result.get('method', 'Unknown')}\n")
                    f.write(f"Task: {result.get('task', 'Unknown')}\n")
                    f.write(f"Throughput: {result.get('throughput', 0)}\n")
                    f.write(f"Latency: {result.get('latency', 0)}\n")
                    f.write("-" * 20 + "\n")
    def visualize_detailed_metrics(self, detailed_metrics: Dict[str, Any], save_path: Optional[Path] = None) -> Path:
        """可视化详细指标"""
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available, skipping detailed metrics visualization")
            if save_path is None:
                save_path = self.output_dir / "detailed_metrics.txt"
            
            with open(save_path, 'w') as f:
                f.write("Detailed Metrics Analysis\n")
                f.write("=" * 30 + "\n")
                f.write(json.dumps(detailed_metrics, indent=2, ensure_ascii=False))
            return save_path
        
        if save_path is None:
            save_path = self.output_dir / "detailed_metrics.png"
        
        # 创建大图表
        fig = plt.figure(figsize=(20, 16))
        
        # 1. 参数效率指标
        if 'parameter_efficiency' in detailed_metrics:
            param_eff = detailed_metrics['parameter_efficiency']
            ax1 = plt.subplot(3, 3, 1)
            
            # 参数数量对比
            labels = ['Total Params', 'Trainable Params']
            values = [param_eff.get('total_parameters', 0), param_eff.get('trainable_parameters', 0)]
            colors = ['lightblue', 'orange']
            
            bars = ax1.bar(labels, values, color=colors)
            ax1.set_title('Parameter Efficiency')
            ax1.set_ylabel('Number of Parameters')
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:,}', ha='center', va='bottom')
        
        # 2. 资源消耗指标
        if 'resource_consumption' in detailed_metrics:
            resource = detailed_metrics['resource_consumption']
            ax2 = plt.subplot(3, 3, 2)
            
            # FLOPs和内存使用
            metrics = ['FLOPs (M)', 'Memory (MB)', 'Latency (ms)']
            values = [
                resource.get('flops_m', 0),
                resource.get('memory_usage_mb', 0),
                resource.get('inference_latency_ms', 0)
            ]
            
            bars = ax2.bar(metrics, values, color=['green', 'red', 'purple'])
            ax2.set_title('Resource Consumption')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # 3. 动态性指标
        if 'dynamic_metrics' in detailed_metrics:
            dynamic = detailed_metrics['dynamic_metrics']
            ax3 = plt.subplot(3, 3, 3)
            
            # Rank统计
            if 'rank_statistics' in dynamic:
                rank_stats = dynamic['rank_statistics']
                rank_values = [
                    rank_stats.get('mean_rank', 0),
                    rank_stats.get('std_rank', 0),
                    rank_stats.get('max_rank', 0),
                    rank_stats.get('min_rank', 0)
                ]
                rank_labels = ['Mean', 'Std', 'Max', 'Min']
                
                ax3.bar(rank_labels, rank_values, color='skyblue')
                ax3.set_title('Rank Statistics')
                ax3.set_ylabel('Rank Value')
        
        # 4. 收敛性指标
        if 'convergence_stability' in detailed_metrics:
            conv = detailed_metrics['convergence_stability']
            ax4 = plt.subplot(3, 3, 4)
            
            # 收敛指标
            conv_metrics = []
            conv_values = []
            
            if 'convergence_step_90' in conv:
                conv_metrics.append('90% Conv')
                conv_values.append(conv['convergence_step_90'])
            if 'convergence_step_95' in conv:
                conv_metrics.append('95% Conv')
                conv_values.append(conv['convergence_step_95'])
            if 'convergence_speed' in conv:
                conv_metrics.append('Speed')
                conv_values.append(conv['convergence_speed'])
            
            if conv_metrics:
                ax4.bar(conv_metrics, conv_values, color='lightgreen')
                ax4.set_title('Convergence Metrics')
                ax4.tick_params(axis='x', rotation=45)
        
        # 5. 可解释性指标
        if 'interpretability' in detailed_metrics:
            interp = detailed_metrics['interpretability']
            ax5 = plt.subplot(3, 3, 5)
            
            # Rank分布
            if 'rank_distribution' in interp:
                rank_dist = interp['rank_distribution']
                if 'effective_rank' in rank_dist and 'nominal_rank' in rank_dist:
                    ranks = ['Nominal', 'Effective']
                    values = [rank_dist['nominal_rank'], rank_dist['effective_rank']]
                    
                    ax5.bar(ranks, values, color=['orange', 'blue'])
                    ax5.set_title('Rank Distribution')
                    ax5.set_ylabel('Rank Value')
        
        # 6. 泛化性指标
        if 'generalization' in detailed_metrics:
            gen = detailed_metrics['generalization']
            ax6 = plt.subplot(3, 3, 6)
            
            # 泛化指标
            gen_metrics = []
            gen_values = []
            
            if 'model_complexity' in gen:
                gen_metrics.append('Complexity')
                gen_values.append(gen['model_complexity'])
            if 'parameter_efficiency' in gen:
                gen_metrics.append('Param Eff')
                gen_values.append(gen['parameter_efficiency'])
            if 'generalization_score' in gen:
                gen_metrics.append('Gen Score')
                gen_values.append(gen['generalization_score'])
            
            if gen_metrics:
                ax6.bar(gen_metrics, gen_values, color='lightcoral')
                ax6.set_title('Generalization Metrics')
                ax6.tick_params(axis='x', rotation=45)
        
        # 7. 训练历史曲线（如果有的话）
        if 'training_history' in detailed_metrics:
            history = detailed_metrics['training_history']
            ax7 = plt.subplot(3, 3, 7)
            
            if isinstance(history, dict) and 'eval_accuracy' in history:
                steps = list(range(len(history['eval_accuracy'])))
                ax7.plot(steps, history['eval_accuracy'], 'b-', label='Accuracy')
                ax7.set_title('Training History')
                ax7.set_xlabel('Steps')
                ax7.set_ylabel('Accuracy')
                ax7.legend()
                ax7.grid(True, alpha=0.3)
        
        # 8. 性能效率散点图
        ax8 = plt.subplot(3, 3, 8)
        
        # 提取关键指标进行效率分析
        accuracy = detailed_metrics.get('task_performance', {}).get('eval_accuracy', 0)
        params = detailed_metrics.get('parameter_efficiency', {}).get('trainable_parameters', 1)
        flops = detailed_metrics.get('resource_consumption', {}).get('flops_m', 1)
        
        if accuracy > 0 and params > 0:
            efficiency = accuracy / (params / 1e6)  # 每百万参数的准确率
            ax8.scatter([params / 1e6], [accuracy], s=100, c='red', alpha=0.7)
            ax8.set_xlabel('Trainable Parameters (M)')
            ax8.set_ylabel('Accuracy')
            ax8.set_title('Parameter Efficiency')
            ax8.grid(True, alpha=0.3)
        
        # 9. 综合评分雷达图
        ax9 = plt.subplot(3, 3, 9, projection='polar')
        
        # 计算各维度评分（0-1标准化）
        categories = []
        scores = []
        
        if 'task_performance' in detailed_metrics:
            categories.append('Performance')
            scores.append(detailed_metrics['task_performance'].get('eval_accuracy', 0))
        
        if 'parameter_efficiency' in detailed_metrics:
            param_eff = detailed_metrics['parameter_efficiency']
            if param_eff.get('total_parameters', 0) > 0:
                categories.append('Efficiency')
                eff_score = 1 - (param_eff.get('trainable_parameters', 0) / param_eff.get('total_parameters', 1))
                scores.append(min(1.0, max(0.0, eff_score)))
        
        if 'resource_consumption' in detailed_metrics:
            categories.append('Resource')
            # 简单的资源评分（越低越好，这里用1-normalized_value）
            mem_score = 1 - min(1.0, detailed_metrics['resource_consumption'].get('memory_usage_mb', 0) / 1000)
            scores.append(max(0.0, mem_score))
        
        if len(categories) >= 3:
            # 闭合雷达图
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            scores += scores[:1]  # 闭合
            angles += angles[:1]  # 闭合
            
            ax9.plot(angles, scores, 'o-', linewidth=2, color='blue')
            ax9.fill(angles, scores, alpha=0.25, color='blue')
            ax9.set_xticks(angles[:-1])
            ax9.set_xticklabels(categories)
            ax9.set_ylim(0, 1)
            ax9.set_title('Comprehensive Score')
        
        plt.suptitle('Detailed Metrics Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
        
        if save_path is None:
            save_path = self.output_dir / "performance_comparison.png"
        
        # 提取性能数据
        data = []
        for result in results:
            if 'benchmarks' in result and 'performance' in result['benchmarks']:
                perf = result['benchmarks']['performance']
                data.append({
                    'method': result['method'],
                    'task': result['task'],
                    'throughput': perf.get('throughput', 0),
                    'latency': perf.get('avg_latency', 0),
                    'memory_peak': perf.get('memory_peak_mb', 0)
                })
        
        if not data:
            return save_path
        
        df = pd.DataFrame(data)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Benchmark Comparison', fontsize=16)
        
        # 吞吐量对比
        sns.barplot(data=df, x='method', y='throughput', hue='task', ax=axes[0, 0])
        axes[0, 0].set_title('Throughput (samples/sec)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 延迟对比
        sns.barplot(data=df, x='method', y='latency', hue='task', ax=axes[0, 1])
        axes[0, 1].set_title('Average Latency (ms)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 内存使用对比
        sns.barplot(data=df, x='method', y='memory_peak', hue='task', ax=axes[1, 0])
        axes[1, 0].set_title('Peak Memory Usage (MB)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 性能效率散点图
        if len(df) > 1:
            sns.scatterplot(data=df, x='latency', y='throughput', 
                          hue='method', size='memory_peak', ax=axes[1, 1])
            axes[1, 1].set_title('Performance Efficiency')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_memory_timeline(self, memory_data: List[Dict], save_path: Optional[Path] = None) -> Path:
        """可视化内存使用时间线"""
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available, skipping memory timeline visualization")
            # Create a simple text file with results
            if save_path is None:
                save_path = self.output_dir / "memory_timeline.txt"
            
            with open(save_path, 'w') as f:
                f.write("Memory Timeline Data\n")
                f.write("=" * 30 + "\n")
                for data in memory_data:
                    f.write(f"Timestamp: {data.get('timestamp', 'Unknown')}\n")
                    f.write(f"Memory Used: {data.get('memory_used', 0)} MB\n")
                    f.write(f"Memory Cached: {data.get('memory_cached', 0)} MB\n")
                    f.write("-" * 20 + "\n")
            return save_path
        
        if save_path is None:
            save_path = self.output_dir / "memory_timeline.png"
        
        if not memory_data:
            return save_path
        
        # 转换数据
        timestamps = [item['timestamp'] for item in memory_data]
        memory_used = [item['memory_used_mb'] for item in memory_data]
        memory_cached = [item.get('memory_cached_mb', 0) for item in memory_data]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, memory_used, label='Memory Used', linewidth=2)
        plt.plot(timestamps, memory_cached, label='Memory Cached', linewidth=2, alpha=0.7)
        plt.fill_between(timestamps, memory_used, alpha=0.3)
        
        plt.title('Memory Usage Timeline')
        plt.xlabel('Time')
        plt.ylabel('Memory (MB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_energy_consumption(self, results: List[Dict], save_path: Optional[Path] = None) -> Path:
        """可视化能耗分析"""
        if not MATPLOTLIB_AVAILABLE:
            logging.warning("Matplotlib not available, skipping energy consumption visualization")
            # Create a simple text file with results
            if save_path is None:
                save_path = self.output_dir / "energy_consumption.txt"
            
            with open(save_path, 'w') as f:
                f.write("Energy Consumption Analysis\n")
                f.write("=" * 35 + "\n")
                for result in results:
                    f.write(f"Method: {result.get('method', 'Unknown')}\n")
                    f.write(f"Task: {result.get('task', 'Unknown')}\n")
                    f.write(f"Total Energy: {result.get('total_energy', 0)} J\n")
                    f.write(f"Average Power: {result.get('avg_power', 0)} W\n")
                    f.write(f"Peak Power: {result.get('peak_power', 0)} W\n")
                    f.write("-" * 20 + "\n")
            return save_path
        
        if save_path is None:
            save_path = self.output_dir / "energy_consumption.png"
        
        # 提取能耗数据
        data = []
        for result in results:
            if 'profiling' in result and 'energy_metrics' in result['profiling']:
                energy = result['profiling']['energy_metrics']
                data.append({
                    'method': result['method'],
                    'task': result['task'],
                    'total_energy': energy.get('total_energy_j', 0),
                    'avg_power': energy.get('avg_power_w', 0),
                    'peak_power': energy.get('peak_power_w', 0)
                })
        
        if not data:
            return save_path
        
        df = pd.DataFrame(data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Energy Consumption Analysis', fontsize=16)
        
        # 总能耗
        sns.barplot(data=df, x='method', y='total_energy', hue='task', ax=axes[0])
        axes[0].set_title('Total Energy (J)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # 平均功率
        sns.barplot(data=df, x='method', y='avg_power', hue='task', ax=axes[1])
        axes[1].set_title('Average Power (W)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # 峰值功率
        sns.barplot(data=df, x='method', y='peak_power', hue='task', ax=axes[2])
        axes[2].set_title('Peak Power (W)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def create_interactive_dashboard(self, results: List[Dict], save_path: Optional[Path] = None) -> Path:
        """创建交互式仪表板"""
        if not PLOTLY_AVAILABLE:
            logging.warning("Plotly not available, skipping interactive dashboard creation")
            # Create a simple HTML file with message
            if save_path is None:
                save_path = self.output_dir / "interactive_dashboard.html"
            
            with open(save_path, 'w') as f:
                f.write("""
                <html>
                <head><title>Benchmark Dashboard</title></head>
                <body>
                    <h1>Interactive Dashboard Not Available</h1>
                    <p>Plotly is not installed. Please install plotly to view interactive dashboard.</p>
                </body>
                </html>
                """)
            return save_path
        
        if save_path is None:
            save_path = self.output_dir / "interactive_dashboard.html"
        
        # 准备数据
        performance_data = []
        memory_data = []
        energy_data = []
        
        for result in results:
            method = result['method']
            task = result['task']
            
            # 性能数据
            if 'benchmarks' in result and 'performance' in result['benchmarks']:
                perf = result['benchmarks']['performance']
                performance_data.append({
                    'Method': method,
                    'Task': task,
                    'Throughput': perf.get('throughput', 0),
                    'Latency': perf.get('avg_latency', 0),
                    'Memory Peak': perf.get('memory_peak_mb', 0)
                })
            
            # 内存数据
            if 'benchmarks' in result and 'memory' in result['benchmarks']:
                mem = result['benchmarks']['memory']
                memory_data.append({
                    'Method': method,
                    'Task': task,
                    'Peak Memory': mem.get('peak_memory_mb', 0),
                    'Avg Memory': mem.get('avg_memory_mb', 0),
                    'Memory Efficiency': mem.get('memory_efficiency', 0)
                })
            
            # 能耗数据
            if 'profiling' in result and 'energy_metrics' in result['profiling']:
                energy = result['profiling']['energy_metrics']
                energy_data.append({
                    'Method': method,
                    'Task': task,
                    'Total Energy': energy.get('total_energy_j', 0),
                    'Avg Power': energy.get('avg_power_w', 0),
                    'Peak Power': energy.get('peak_power_w', 0)
                })
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Memory Usage', 
                          'Energy Consumption', 'Efficiency Analysis'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        if performance_data:
            df_perf = pd.DataFrame(performance_data)
            
            # 性能图表
            for method in df_perf['Method'].unique():
                method_data = df_perf[df_perf['Method'] == method]
                fig.add_trace(
                    go.Scatter(x=method_data['Task'], y=method_data['Throughput'],
                             mode='markers+lines', name=f'{method} Throughput',
                             marker=dict(size=method_data['Memory Peak']/10)),
                    row=1, col=1
                )
        
        if memory_data:
            df_mem = pd.DataFrame(memory_data)
            
            # 内存图表
            fig.add_trace(
                go.Bar(x=df_mem['Method'], y=df_mem['Peak Memory'],
                      name='Peak Memory', opacity=0.7),
                row=1, col=2
            )
        
        if energy_data:
            df_energy = pd.DataFrame(energy_data)
            
            # 能耗图表
            fig.add_trace(
                go.Scatter(x=df_energy['Method'], y=df_energy['Total Energy'],
                         mode='markers', name='Total Energy',
                         marker=dict(size=df_energy['Peak Power']/2)),
                row=2, col=1
            )
        
        # 效率分析
        if performance_data and energy_data:
            efficiency_data = []
            for perf, energy in zip(performance_data, energy_data):
                if perf['Method'] == energy['Method'] and perf['Task'] == energy['Task']:
                    efficiency = perf['Throughput'] / max(energy['Total Energy'], 1)
                    efficiency_data.append({
                        'Method': perf['Method'],
                        'Task': perf['Task'],
                        'Efficiency': efficiency
                    })
            
            if efficiency_data:
                df_eff = pd.DataFrame(efficiency_data)
                fig.add_trace(
                    go.Bar(x=df_eff['Method'], y=df_eff['Efficiency'],
                          name='Energy Efficiency'),
                    row=2, col=2
                )
        
        # 更新布局
        fig.update_layout(
            title_text="LoRAven Benchmark Dashboard",
            showlegend=True,
            height=800
        )
        
        # 保存HTML文件
        fig.write_html(save_path)
        
        return save_path


class ReportGenerator:
    """基准测试报告生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_comprehensive_report(self, results: List[Dict], 
                                    visualizations: Dict[str, Path]) -> Path:
        """生成综合报告"""
        report_path = self.output_dir / "benchmark_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# LoRAven Benchmark Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 执行摘要
            f.write("## Executive Summary\n\n")
            f.write(f"Total experiments: {len(results)}\n")
            
            methods = set(r['method'] for r in results)
            tasks = set(r['task'] for r in results)
            f.write(f"Methods tested: {', '.join(methods)}\n")
            f.write(f"Tasks evaluated: {', '.join(tasks)}\n\n")
            
            # 性能概览
            f.write("## Performance Overview\n\n")
            self._write_performance_summary(f, results)
            
            # 详细结果
            f.write("## Detailed Results\n\n")
            for result in results:
                self._write_experiment_details(f, result)
            
            # 可视化
            f.write("## Visualizations\n\n")
            for name, path in visualizations.items():
                f.write(f"### {name.replace('_', ' ').title()}\n")
                f.write(f"![{name}]({path.name})\n\n")
            
            # 结论和建议
            f.write("## Conclusions and Recommendations\n\n")
            self._write_conclusions(f, results)
        
        return report_path
    
    def _write_performance_summary(self, f, results: List[Dict]):
        """写入性能摘要"""
        # 收集性能指标
        throughputs = []
        latencies = []
        memory_peaks = []
        
        for result in results:
            if 'benchmarks' in result and 'performance' in result['benchmarks']:
                perf = result['benchmarks']['performance']
                throughputs.append(perf.get('throughput', 0))
                latencies.append(perf.get('avg_latency', 0))
                memory_peaks.append(perf.get('memory_peak_mb', 0))
        
        if throughputs:
            f.write(f"- Average throughput: {np.mean(throughputs):.2f} samples/sec\n")
            f.write(f"- Average latency: {np.mean(latencies):.2f} ms\n")
            f.write(f"- Average peak memory: {np.mean(memory_peaks):.2f} MB\n\n")
    
    def _write_experiment_details(self, f, result: Dict):
        """写入实验详情"""
        f.write(f"### {result['method']} on {result['task']}\n\n")
        
        # 基本信息
        f.write(f"- Timestamp: {result['timestamp']}\n")
        f.write(f"- Status: {result['status']}\n")
        
        # 准确性指标
        if 'metrics' in result and 'accuracy' in result['metrics']:
            acc = result['metrics']['accuracy']
            f.write(f"- Accuracy: {acc:.4f}\n")
        
        # 性能指标
        if 'benchmarks' in result:
            benchmarks = result['benchmarks']
            for bench_name, bench_data in benchmarks.items():
                f.write(f"- {bench_name.title()} Benchmark:\n")
                if isinstance(bench_data, dict):
                    for key, value in bench_data.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  - {key}: {value:.4f}\n")
        
        f.write("\n")
    
    def _write_conclusions(self, f, results: List[Dict]):
        """写入结论和建议"""
        f.write("Based on the benchmark results:\n\n")
        
        # 找出最佳性能方法
        best_throughput = None
        best_method = None
        
        for result in results:
            if 'benchmarks' in result and 'performance' in result['benchmarks']:
                throughput = result['benchmarks']['performance'].get('throughput', 0)
                if best_throughput is None or throughput > best_throughput:
                    best_throughput = throughput
                    best_method = result['method']
        
        if best_method:
            f.write(f"- **{best_method}** shows the best throughput performance\n")
        
        f.write("- Memory usage varies significantly across methods\n")
        f.write("- Energy efficiency should be considered for production deployment\n")
        f.write("- Further optimization may be needed for specific use cases\n\n")


def create_benchmark_report(results_dir: Path, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """创建完整的基准测试报告"""
    if output_dir is None:
        output_dir = results_dir / "reports"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载结果
    results = []
    for result_file in results_dir.glob("*.json"):
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
                results.append(result)
        except Exception as e:
            print(f"Failed to load {result_file}: {e}")
    
    if not results:
        print("No results found to visualize")
        return {}
    
    # 创建可视化
    visualizer = BenchmarkVisualizer(output_dir)
    
    visualizations = {}
    
    # 性能对比
    perf_path = visualizer.visualize_performance_comparison(results)
    visualizations['performance_comparison'] = perf_path
    
    # 内存时间线（如果有数据）
    for result in results:
        if 'profiling' in result and 'memory_timeline' in result['profiling']:
            memory_path = visualizer.visualize_memory_timeline(
                result['profiling']['memory_timeline']
            )
            visualizations['memory_timeline'] = memory_path
            break
    
    # 能耗分析
    energy_path = visualizer.visualize_energy_consumption(results)
    visualizations['energy_consumption'] = energy_path
    
    # 详细指标可视化（新增）
    for result in results:
        if 'detailed_metrics' in result:
            method = result.get('method', 'unknown')
            task = result.get('task', 'unknown')
            detailed_path = visualizer.visualize_detailed_metrics(
                result['detailed_metrics'], 
                output_dir / f"detailed_metrics_{method}_{task}.png"
            )
            visualizations[f'detailed_metrics_{method}_{task}'] = detailed_path
    
    # 交互式仪表板
    dashboard_path = visualizer.create_interactive_dashboard(results)
    visualizations['interactive_dashboard'] = dashboard_path
    
    # 生成报告
    report_generator = ReportGenerator(output_dir)
    report_path = report_generator.generate_comprehensive_report(results, visualizations)
    visualizations['comprehensive_report'] = report_path
    
    return visualizations