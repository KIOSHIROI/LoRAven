"""
Matplotlib 工具模块：解决中文显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os
from typing import List, Optional


def setup_chinese_font(force_font: Optional[str] = None) -> bool:
    """
    设置中文字体支持
    
    Args:
        force_font: 强制使用指定字体名称
        
    Returns:
        success: 是否成功设置中文字体
    """
    try:
        if force_font:
            plt.rcParams['font.sans-serif'] = [force_font, 'DejaVu Sans', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"强制使用字体: {force_font}")
            return True
        
        # 根据操作系统选择合适的中文字体
        system = platform.system()
        if system == "Darwin":  # macOS
            chinese_fonts = [
                'PingFang SC',
                'Hiragino Sans GB',
                'STHeiti',
                'Arial Unicode MS',
                'Noto Sans CJK SC',
                'Noto Serif CJK SC'
            ]
        elif system == "Windows":
            chinese_fonts = [
                'Microsoft YaHei',
                'SimHei',
                'SimSun',
                'KaiTi',
                'FangSong',
                'Arial Unicode MS'
            ]
        else:  # Linux
            chinese_fonts = [
                'Noto Sans CJK SC',
                'Noto Sans CJK JP',
                'Noto Serif CJK SC',
                'Noto Serif CJK JP',
                'AR PL UMing CN',
                'AR PL UKai CN',
                'WenQuanYi Micro Hei',
                'WenQuanYi Zen Hei',
                'DejaVu Sans',
                'Arial Unicode MS'
            ]
        
        # 获取系统中可用的字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 查找可用的中文字体
        found_fonts = []
        for font in chinese_fonts:
            if font in available_fonts:
                found_fonts.append(font)
        
        if found_fonts:
            # 设置字体优先级列表，包含回退字体
            font_list = found_fonts + ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['font.sans-serif'] = font_list
            plt.rcParams['axes.unicode_minus'] = False
            print(f"成功设置中文字体: {found_fonts[0]} (共找到 {len(found_fonts)} 个中文字体)")
            return True
        
        # 如果没有找到中文字体，使用通用字体
        print("警告: 未找到专门的中文字体，使用通用字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        return False
        
    except Exception as e:
        print(f"字体设置失败: {e}")
        # 设置基本字体配置
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return False


def try_install_chinese_fonts():
    """尝试安装中文字体（Linux 系统）"""
    try:
        import subprocess
        import urllib.request
        
        print("尝试安装中文字体...")
        
        # 检查是否已安装 Noto 字体
        result = subprocess.run(['fc-list', ':lang=zh'], capture_output=True, text=True)
        if 'Noto' in result.stdout:
            print("已找到 Noto 中文字体")
            return True
        
        # 尝试安装 Noto 字体
        print("正在安装 Noto 中文字体...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=False)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'fonts-noto-cjk'], check=False)
        
        # 清除字体缓存
        subprocess.run(['fc-cache', '-fv'], check=False)
        
        print("中文字体安装完成，请重新运行程序")
        return True
        
    except Exception as e:
        print(f"字体安装失败: {e}")
        return False


def get_available_chinese_fonts() -> List[str]:
    """
    获取系统中可用的中文字体
    
    Returns:
        fonts: 可用中文字体列表
    """
    try:
        # 获取所有字体
        all_fonts = [f.name for f in fm.fontManager.ttflist]
        
        # 过滤中文字体
        chinese_keywords = [
            'chinese', 'cjk', 'noto', 'pingfang', 'hiragino', 'stheiti',
            'microsoft', 'simhei', 'simsun', 'kaiti', 'fangsong',
            'uming', 'ukai', 'wenquanyi', 'dejavu'
        ]
        
        chinese_fonts = []
        for font in all_fonts:
            if any(keyword in font.lower() for keyword in chinese_keywords):
                chinese_fonts.append(font)
        
        return sorted(list(set(chinese_fonts)))
        
    except Exception as e:
        print(f"获取字体列表失败: {e}")
        return []


def test_chinese_display():
    """测试中文显示效果"""
    import numpy as np
    
    # 创建测试图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, label='正弦波')
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('幅度')
    ax.set_title('中文显示测试')
    ax.legend()
    ax.grid(True)
    
    # 添加中文文本
    ax.text(5, 0.5, '这是中文测试文本', fontsize=14, ha='center')
    
    plt.tight_layout()
    plt.savefig('chinese_font_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("中文显示测试完成，图片已保存为 'chinese_font_test.png'")


def configure_matplotlib_for_chinese():
    """配置 matplotlib 以支持中文显示"""
    # 设置中文字体
    success = setup_chinese_font()
    
    if not success:
        print("中文字体设置失败，将使用英文标签")
        return False
    
    # 设置其他参数
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    return True


def force_chinese_font_fix():
    """强制修复中文字体显示问题"""
    try:
        # 清除字体缓存
        fm._rebuild()
        
        # 设置多种字体回退
        font_candidates = [
            'Noto Sans CJK SC',
            'Noto Sans CJK JP', 
            'Noto Serif CJK SC',
            'Noto Serif CJK JP',
            'AR PL UMing CN',
            'AR PL UKai CN',
            'DejaVu Sans',
            'Arial Unicode MS',
            'sans-serif'
        ]
        
        # 获取可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        working_fonts = [f for f in font_candidates if f in available_fonts]
        
        if working_fonts:
            plt.rcParams['font.sans-serif'] = working_fonts
            plt.rcParams['axes.unicode_minus'] = False
            print(f"强制字体修复成功，使用字体: {working_fonts[0]}")
            return True
        else:
            # 使用系统默认字体
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            print("使用系统默认字体")
            return False
            
    except Exception as e:
        print(f"强制字体修复失败: {e}")
        return False


if __name__ == '__main__':
    # 测试中文字体设置
    print("=== Matplotlib 中文字体设置测试 ===")
    
    # 显示可用中文字体
    fonts = get_available_chinese_fonts()
    print(f"系统中可用的中文字体: {fonts[:5]}...")
    
    # 设置中文字体
    success = setup_chinese_font()
    
    if success:
        print("✅ 中文字体设置成功")
        # 运行显示测试
        test_chinese_display()
    else:
        print("❌ 中文字体设置失败")
        print("建议手动安装中文字体或使用英文标签")
