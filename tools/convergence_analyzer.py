# 收敛性分析器
# 功能：分析训练收敛性和稳定性，包括收敛步数、种子稳定性等指标

import numpy as np
import torch
from typing import Dict, List, Optional, Any, Tuple
import logging
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConvergenceAnalyzer:
    """收敛性和稳定性分析器"""
    
    def __init__(self):
        self.convergence_history = []
        self.seed_results = defaultdict(list)
        
    def analyze_convergence_stability(self, training_history: List[Dict], 
                                    target_metric: str = 'eval_accuracy') -> Dict[str, Any]:
        """分析收敛性和稳定性"""
        if not training_history:
            return {}
        
        # 提取目标指标历史
        metric_history = []
        step_history = []
        
        for i, step_data in enumerate(training_history):
            if target_metric in step_data:
                metric_history.append(step_data[target_metric])
                step_history.append(i + 1)
        
        if not metric_history:
            return {}
        
        # 计算收敛指标
        convergence_metrics = self._calculate_convergence_metrics(
            metric_history, step_history, target_metric
        )
        
        # 计算稳定性指标
        stability_metrics = self._calculate_stability_metrics(metric_history)
        
        # 计算学习曲线特征
        learning_curve_features = self._analyze_learning_curve(metric_history)
        
        return {
            'convergence_metrics': convergence_metrics,
            'stability_metrics': stability_metrics,
            'learning_curve_features': learning_curve_features,
            'metric_name': target_metric,
            'total_steps': len(metric_history)
        }
    
    def _calculate_convergence_metrics(self, metric_history: List[float], 
                                     step_history: List[int],
                                     target_metric: str) -> Dict[str, Any]:
        """计算收敛指标"""
        metrics = {}
        
        if not metric_history:
            return metrics
        
        # 找到最佳性能
        best_metric = max(metric_history)
        best_step = step_history[np.argmax(metric_history)]
        
        # 计算达到95%最佳性能的步数
        target_95 = best_metric * 0.95
        steps_to_95 = None
        
        for i, metric in enumerate(metric_history):
            if metric >= target_95:
                steps_to_95 = step_history[i]
                break
        
        # 计算达到90%最佳性能的步数
        target_90 = best_metric * 0.90
        steps_to_90 = None
        
        for i, metric in enumerate(metric_history):
            if metric >= target_90:
                steps_to_90 = step_history[i]
                break
        
        # 计算收敛速度 (前半段的平均提升)
        mid_point = len(metric_history) // 2
        if mid_point > 0:
            early_improvement = (metric_history[mid_point] - metric_history[0]) / mid_point
        else:
            early_improvement = 0.0
        
        # 计算后期稳定性 (后半段的方差)
        if len(metric_history) > mid_point:
            late_stability = np.var(metric_history[mid_point:])
        else:
            late_stability = 0.0
        
        metrics.update({
            'best_metric': best_metric,
            'best_step': best_step,
            'steps_to_95_percent_best': steps_to_95 or len(step_history),
            'steps_to_90_percent_best': steps_to_90 or len(step_history),
            'convergence_speed': early_improvement,
            'late_stage_stability': late_stability,
            'final_metric': metric_history[-1],
            'improvement_ratio': (metric_history[-1] - metric_history[0]) / metric_history[0] if metric_history[0] != 0 else 0.0
        })
        
        return metrics
    
    def _calculate_stability_metrics(self, metric_history: List[float]) -> Dict[str, Any]:
        """计算稳定性指标"""
        metrics = {}
        
        if len(metric_history) < 2:
            return metrics
        
        # 基本统计指标
        mean_metric = np.mean(metric_history)
        std_metric = np.std(metric_history)
        var_metric = np.var(metric_history)
        
        # 变异系数 (相对稳定性)
        cv = std_metric / mean_metric if mean_metric != 0 else 0.0
        
        # 计算趋势稳定性 (线性回归的R²)
        x = np.arange(len(metric_history))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, metric_history)
        trend_stability = r_value ** 2
        
        # 计算局部稳定性 (相邻点的平均变化)
        local_changes = np.abs(np.diff(metric_history))
        avg_local_change = np.mean(local_changes)
        max_local_change = np.max(local_changes)
        
        # 计算单调性 (是否单调递增)
        monotonic_increases = sum(1 for i in range(1, len(metric_history)) 
                                if metric_history[i] >= metric_history[i-1])
        monotonicity_ratio = monotonic_increases / (len(metric_history) - 1) if len(metric_history) > 1 else 0.0
        
        metrics.update({
            'mean_metric': mean_metric,
            'std_metric': std_metric,
            'variance_metric': var_metric,
            'coefficient_of_variation': cv,
            'trend_stability_r2': trend_stability,
            'trend_slope': slope,
            'trend_p_value': p_value,
            'avg_local_change': avg_local_change,
            'max_local_change': max_local_change,
            'monotonicity_ratio': monotonicity_ratio,
            'stability_score': 1.0 / (1.0 + cv)  # 综合稳定性评分
        })
        
        return metrics
    
    def _analyze_learning_curve(self, metric_history: List[float]) -> Dict[str, Any]:
        """分析学习曲线特征"""
        features = {}
        
        if len(metric_history) < 3:
            return features
        
        # 分析学习阶段
        total_steps = len(metric_history)
        
        # 早期阶段 (前25%)
        early_end = max(1, total_steps // 4)
        early_metrics = metric_history[:early_end]
        early_improvement = (early_metrics[-1] - early_metrics[0]) / len(early_metrics) if len(early_metrics) > 1 else 0.0
        
        # 中期阶段 (25%-75%)
        mid_start = early_end
        mid_end = max(mid_start + 1, 3 * total_steps // 4)
        mid_metrics = metric_history[mid_start:mid_end]
        mid_improvement = (mid_metrics[-1] - mid_metrics[0]) / len(mid_metrics) if len(mid_metrics) > 1 else 0.0
        
        # 后期阶段 (后25%)
        late_start = mid_end
        late_metrics = metric_history[late_start:]
        late_improvement = (late_metrics[-1] - late_metrics[0]) / len(late_metrics) if len(late_metrics) > 1 else 0.0
        
        # 检测过拟合 (性能下降)
        peak_idx = np.argmax(metric_history)
        peak_value = metric_history[peak_idx]
        final_value = metric_history[-1]
        overfitting_degree = (peak_value - final_value) / peak_value if peak_value != 0 else 0.0
        
        # 检测学习饱和 (后期改善很小)
        saturation_threshold = 0.001  # 0.1%的改善阈值
        is_saturated = abs(late_improvement) < saturation_threshold
        
        features.update({
            'early_stage_improvement': early_improvement,
            'mid_stage_improvement': mid_improvement,
            'late_stage_improvement': late_improvement,
            'peak_step': peak_idx + 1,
            'peak_value': peak_value,
            'overfitting_degree': overfitting_degree,
            'is_overfitting': overfitting_degree > 0.05,  # 5%下降认为过拟合
            'is_saturated': is_saturated,
            'learning_efficiency': (final_value - metric_history[0]) / total_steps if total_steps > 0 else 0.0
        })
        
        return features
    
    def analyze_multi_seed_stability(self, seed_results: Dict[int, List[float]], 
                                   target_metric: str = 'eval_accuracy') -> Dict[str, Any]:
        """分析多种子稳定性"""
        if not seed_results:
            return {}
        
        # 提取每个种子的最终结果
        final_results = []
        best_results = []
        convergence_steps = []
        
        for seed, history in seed_results.items():
            if history:
                final_results.append(history[-1])
                best_results.append(max(history))
                
                # 计算收敛步数 (达到95%最佳性能)
                best_val = max(history)
                target_val = best_val * 0.95
                conv_step = len(history)
                for i, val in enumerate(history):
                    if val >= target_val:
                        conv_step = i + 1
                        break
                convergence_steps.append(conv_step)
        
        if not final_results:
            return {}
        
        # 计算种子间稳定性
        stability_metrics = {
            'num_seeds': len(final_results),
            'final_results': {
                'mean': float(np.mean(final_results)),
                'std': float(np.std(final_results)),
                'min': float(np.min(final_results)),
                'max': float(np.max(final_results)),
                'range': float(np.max(final_results) - np.min(final_results)),
                'cv': float(np.std(final_results) / np.mean(final_results)) if np.mean(final_results) != 0 else 0.0
            },
            'best_results': {
                'mean': float(np.mean(best_results)),
                'std': float(np.std(best_results)),
                'min': float(np.min(best_results)),
                'max': float(np.max(best_results))
            },
            'convergence_stability': {
                'mean_steps': float(np.mean(convergence_steps)),
                'std_steps': float(np.std(convergence_steps)),
                'min_steps': int(np.min(convergence_steps)),
                'max_steps': int(np.max(convergence_steps))
            }
        }
        
        # 计算置信区间
        if len(final_results) > 1:
            confidence_interval = stats.t.interval(
                0.95, len(final_results) - 1,
                loc=np.mean(final_results),
                scale=stats.sem(final_results)
            )
            stability_metrics['confidence_interval_95'] = {
                'lower': float(confidence_interval[0]),
                'upper': float(confidence_interval[1])
            }
        
        return stability_metrics
    
    def generate_convergence_plot(self, training_history: List[Dict], 
                                target_metric: str = 'eval_accuracy',
                                output_path: str = None) -> Optional[str]:
        """生成收敛曲线图"""
        if not training_history:
            return None
        
        # 提取数据
        steps = []
        metrics = []
        
        for i, step_data in enumerate(training_history):
            if target_metric in step_data:
                steps.append(i + 1)
                metrics.append(step_data[target_metric])
        
        if not metrics:
            return None
        
        # 绘制收敛曲线
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(steps, metrics, 'b-', linewidth=2, label=target_metric)
        plt.xlabel('Training Steps')
        plt.ylabel(target_metric.replace('_', ' ').title())
        plt.title('Training Convergence Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 绘制移动平均
        if len(metrics) > 5:
            window_size = min(10, len(metrics) // 5)
            moving_avg = np.convolve(metrics, np.ones(window_size)/window_size, mode='valid')
            moving_steps = steps[window_size-1:]
            
            plt.subplot(2, 2, 2)
            plt.plot(steps, metrics, 'b-', alpha=0.5, label='Original')
            plt.plot(moving_steps, moving_avg, 'r-', linewidth=2, label=f'Moving Avg (window={window_size})')
            plt.xlabel('Training Steps')
            plt.ylabel(target_metric.replace('_', ' ').title())
            plt.title('Smoothed Convergence Curve')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        # 绘制改善率
        if len(metrics) > 1:
            improvements = np.diff(metrics)
            
            plt.subplot(2, 2, 3)
            plt.plot(steps[1:], improvements, 'g-', linewidth=1, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            plt.xlabel('Training Steps')
            plt.ylabel('Step-wise Improvement')
            plt.title('Learning Progress')
            plt.grid(True, alpha=0.3)
        
        # 绘制累积改善
        if len(metrics) > 1:
            cumulative_improvement = np.array(metrics) - metrics[0]
            
            plt.subplot(2, 2, 4)
            plt.plot(steps, cumulative_improvement, 'purple', linewidth=2)
            plt.xlabel('Training Steps')
            plt.ylabel('Cumulative Improvement')
            plt.title('Total Progress from Start')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return output_path
        else:
            plt.show()
            return None
    
    def save_convergence_report(self, analysis_results: Dict[str, Any], 
                              output_path: str):
        """保存收敛性分析报告"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成markdown报告
        report_lines = [
            "# Convergence and Stability Analysis Report",
            "",
            f"## Target Metric: {analysis_results.get('metric_name', 'Unknown')}",
            f"## Total Training Steps: {analysis_results.get('total_steps', 'N/A')}",
            ""
        ]
        
        # 收敛指标
        if 'convergence_metrics' in analysis_results:
            conv = analysis_results['convergence_metrics']
            report_lines.extend([
                "### Convergence Metrics",
                "",
                f"- **Best Performance**: {conv.get('best_metric', 'N/A'):.4f}",
                f"- **Best Step**: {conv.get('best_step', 'N/A')}",
                f"- **Steps to 95% Best**: {conv.get('steps_to_95_percent_best', 'N/A')}",
                f"- **Steps to 90% Best**: {conv.get('steps_to_90_percent_best', 'N/A')}",
                f"- **Convergence Speed**: {conv.get('convergence_speed', 'N/A'):.6f}",
                f"- **Final Performance**: {conv.get('final_metric', 'N/A'):.4f}",
                f"- **Improvement Ratio**: {conv.get('improvement_ratio', 'N/A'):.4f}",
                ""
            ])
        
        # 稳定性指标
        if 'stability_metrics' in analysis_results:
            stab = analysis_results['stability_metrics']
            report_lines.extend([
                "### Stability Metrics",
                "",
                f"- **Mean Performance**: {stab.get('mean_metric', 'N/A'):.4f}",
                f"- **Standard Deviation**: {stab.get('std_metric', 'N/A'):.4f}",
                f"- **Coefficient of Variation**: {stab.get('coefficient_of_variation', 'N/A'):.4f}",
                f"- **Trend Stability (R²)**: {stab.get('trend_stability_r2', 'N/A'):.4f}",
                f"- **Average Local Change**: {stab.get('avg_local_change', 'N/A'):.4f}",
                f"- **Monotonicity Ratio**: {stab.get('monotonicity_ratio', 'N/A'):.4f}",
                f"- **Stability Score**: {stab.get('stability_score', 'N/A'):.4f}",
                ""
            ])
        
        # 学习曲线特征
        if 'learning_curve_features' in analysis_results:
            curve = analysis_results['learning_curve_features']
            report_lines.extend([
                "### Learning Curve Features",
                "",
                f"- **Early Stage Improvement**: {curve.get('early_stage_improvement', 'N/A'):.6f}",
                f"- **Mid Stage Improvement**: {curve.get('mid_stage_improvement', 'N/A'):.6f}",
                f"- **Late Stage Improvement**: {curve.get('late_stage_improvement', 'N/A'):.6f}",
                f"- **Peak Step**: {curve.get('peak_step', 'N/A')}",
                f"- **Peak Value**: {curve.get('peak_value', 'N/A'):.4f}",
                f"- **Overfitting Degree**: {curve.get('overfitting_degree', 'N/A'):.4f}",
                f"- **Is Overfitting**: {curve.get('is_overfitting', 'N/A')}",
                f"- **Is Saturated**: {curve.get('is_saturated', 'N/A')}",
                f"- **Learning Efficiency**: {curve.get('learning_efficiency', 'N/A'):.6f}",
                ""
            ])
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Convergence analysis report saved to {output_path}")