# GLUE完整实验运行脚本
# 该脚本执行LoRA、AdaLoRA、DoRA在GLUE任务上的完整对比实验
# 测试功能：
# 1. 在SST-2、MNLI、QNLI、RTE、CoLA五个任务上运行实验
# 2. 对比LoRA、AdaLoRA、DoRA三种PEFT方法的性能
# 3. 收集并保存所有实验结果
# 4. 生成性能对比报告

import os
import sys
import json
import time
from pathlib import Path

# 添加实验目录到路径
sys.path.append('experiments')
from glue_benchmark import GLUEBenchmark, ExperimentConfig

def main():
    """运行完整的GLUE基准测试实验"""
    
    # 设置Hugging Face镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 设置tokenizers并行化，避免fork警告
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 设置Hugging Face缓存目录到项目本地
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hf_cache_dir = os.path.join(project_root, ".hf_cache")
    
    # 创建缓存目录
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(os.path.join(hf_cache_dir, "transformers"), exist_ok=True)
    os.makedirs(os.path.join(hf_cache_dir, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(hf_cache_dir, "hub"), exist_ok=True)
    
    # 设置环境变量
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache_dir, "hub")
    
    # 实验配置
    config = ExperimentConfig(
        model_name='bert-base-uncased',
        tasks=['sst2', 'mnli', 'qnli', 'rte', 'cola'],
        max_length=128,
        batch_size=100,  # 减小批次大小以节省GPU内存
        num_epochs=1,   # 完整训练3个epoch
        learning_rate=2e-5,
        warmup_steps=500,
        output_dir='./full_experiment_results',
        use_mixed_precision=True,  # 启用混合精度训练
        dataloader_num_workers=8   # 增加数据加载器工作进程
    )
    
    # PEFT方法列表
    methods = ['lora', 'adalora', 'dora']
    
    # 创建基准测试实例
    benchmark = GLUEBenchmark(config)
    
    print("🚀 开始GLUE完整基准测试实验...")
    print(f"任务: {config.tasks}")
    print(f"方法: {methods}")
    print(f"使用设备: CUDA:0, CUDA:1, CUDA:2 ,CUDA:3 (多GPU并行训练)")
    print(f"批次大小: {config.batch_size} (每GPU)")
    print(f"混合精度训练: {'启用' if config.use_mixed_precision else '禁用'}")
    print("=" * 60)
    
    # 运行所有实验
    start_time = time.time()
    
    try:
        results = benchmark.run_all_experiments(methods)
        
        # 保存结果
        results_file = Path(config.output_dir) / "complete_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        benchmark.generate_comparison_report(results)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n" + "=" * 60)
        print("✅ 实验完成!")
        print(f"总耗时: {total_time/3600:.2f} 小时")
        print(f"结果保存在: {results_file}")
        print("详细报告已生成")
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()