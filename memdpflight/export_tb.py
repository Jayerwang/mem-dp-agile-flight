import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ==================== 配置区域 ====================

# TensorBoard 日志目录（单个 run 文件夹）
LOG_DIR = 'runs/Apr25_23-13-38_autodl-container-6ad647b547-612389ca'

# 输出目录
OUTPUT_DIR = 'results_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 可选：指定需要绘制的标签（留空列表则自动绘制所有）
# 例如：SELECTED_TAGS = ['loss', 'success']
SELECTED_TAGS = []  # 空列表 = 绘制所有可用的 scalar

# ==================== 主程序 ====================

# 加载 TensorBoard 数据
ea = event_accumulator.EventAccumulator(LOG_DIR)
ea.Reload()

# 获取所有可用的 scalar 标签
all_tags = ea.Tags()['scalars']

# 确定最终要绘制的标签
tags_to_plot = SELECTED_TAGS if SELECTED_TAGS else all_tags

print(f"发现 {len(all_tags)} 个 scalar 标签: {all_tags}")
print(f"将绘制 {len(tags_to_plot)} 个: {tags_to_plot}\n")

for tag in tags_to_plot:
    # 检查标签是否真实存在
    if tag not in all_tags:
        print(f"警告: 标签 '{tag}' 不存在，已跳过")
        continue
    
    # 提取数据
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    
    # 跳过空数据
    if len(steps) == 0:
        print(f"跳过空数据: {tag}")
        continue
    
    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(
        steps, values,
        label=tag,
        color='#FF5722' if 'loss' in tag.lower() else '#3F51B5',
        linewidth=1.5
    )
    plt.xlabel('Steps', fontsize=12)
    plt.ylabel(tag, fontsize=12)
    plt.title(f'Training Curve: {tag}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best')
    
    # 保存图片
    # 清理文件名中的非法字符
    safe_tag = tag.replace('/', '_').replace('\\', '_')
    save_path = os.path.join(OUTPUT_DIR, f'{safe_tag}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {save_path} ({len(steps)} 个点)")

print(f"\n所有图像已保存至: {os.path.abspath(OUTPUT_DIR)}")