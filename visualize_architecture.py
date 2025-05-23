#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReverberationNet 网状架构可视化工具
生成网状连接图和数据流图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def draw_mesh_architecture():
    """
    绘制网状架构图
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # 标题
    ax.text(50, 95, 'ReverberationNet 网状架构', fontsize=20, 
            ha='center', fontweight='bold', color='navy')
    
    # 角色模块位置（圆形排列）
    role_names = [
        'Eileen', 'Pluto', 'Organ', 'Harp', 'WolfHour', 'Viola', 'Philip',
        'Cello', 'CircusMaster', 'Bremen', 'Zaixian', 'Elena', 'Greta',
        'Clarinet', 'Horn', 'Tuba', 'Trombone', 'Violin1', 'Violin2'
    ]
    
    role_colors = plt.cm.Set3(np.linspace(0, 1, len(role_names)))
    
    # 角色位置（外圈）
    role_positions = {}
    center_x, center_y = 50, 50
    radius = 35
    
    for i, role in enumerate(role_names):
        angle = 2 * np.pi * i / len(role_names)
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        role_positions[role.lower()] = (x, y)
        
        # 绘制角色模块
        circle = plt.Circle((x, y), 2.5, color=role_colors[i], alpha=0.7, ec='black')
        ax.add_patch(circle)
        ax.text(x, y-4, role, fontsize=8, ha='center', fontweight='bold')
    
    # 融合处理器位置（内圈）
    processor_names = ['Harmony', 'Rhythm', 'Melody', 'Texture', 'Dynamics', 'Timbre', 'Structure']
    processor_colors = plt.cm.Set1(np.linspace(0, 1, len(processor_names)))
    
    processor_positions = {}
    proc_radius = 15
    
    for i, processor in enumerate(processor_names):
        angle = 2 * np.pi * i / len(processor_names)
        x = center_x + proc_radius * np.cos(angle)
        y = center_y + proc_radius * np.sin(angle)
        processor_positions[processor.lower()] = (x, y)
        
        # 绘制融合处理器
        rect = FancyBboxPatch((x-4, y-2), 8, 4, boxstyle="round,pad=0.2",
                             facecolor=processor_colors[i], alpha=0.8, 
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, processor, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Argallia指挥层（中心）
    argallia_circle = plt.Circle((center_x, center_y), 5, color='gold', alpha=0.9, ec='red', linewidth=2)
    ax.add_patch(argallia_circle)
    ax.text(center_x, center_y, 'Argallia\n指挥层', fontsize=10, ha='center', va='center', fontweight='bold')
    
    # 绘制连接线
    try:
        from ReverbNet import ReverberationNet
        model = ReverberationNet(d=64, num_instruments=7)
        structure_info = model.get_network_structure()
        
        # 角色→融合处理器连接（红色）
        for role_name, processor_name in structure_info['role_to_processor'].items():
            if role_name in role_positions and processor_name in processor_positions:
                x1, y1 = role_positions[role_name]
                x2, y2 = processor_positions[processor_name]
                ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.6, linewidth=1.5)
        
        # 融合处理器→角色反馈连接（蓝色）
        for processor_name, connected_roles in structure_info['processor_to_roles'].items():
            if processor_name in processor_positions:
                px, py = processor_positions[processor_name]
                for role_name in connected_roles:
                    if role_name in role_positions:
                        rx, ry = role_positions[role_name]
                        ax.plot([px, rx], [py, ry], 'b--', alpha=0.4, linewidth=1)
        
        # 所有到Argallia的连接（绿色）
        for pos in list(role_positions.values()) + list(processor_positions.values()):
            ax.plot([pos[0], center_x], [pos[1], center_y], 'g:', alpha=0.3, linewidth=0.8)
            
    except ImportError:
        print("无法导入ReverbNet，跳过连接线绘制")
    
    # 图例
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=2, label='角色→融合处理器'),
        plt.Line2D([0], [0], color='blue', lw=2, linestyle='--', label='融合处理器→角色反馈'),
        plt.Line2D([0], [0], color='green', lw=2, linestyle=':', label='→Argallia汇聚'),
        plt.Circle((0, 0), 0.1, color='lightblue', label='角色模块'),
        patches.Rectangle((0, 0), 0.1, 0.1, color='orange', label='融合处理器'),
        plt.Circle((0, 0), 0.1, color='gold', label='Argallia指挥层')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # 保存图片
    plt.tight_layout()
    plt.savefig('reverbnet_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ 网状架构图已保存到: reverbnet_architecture.png")
    
    return fig

def print_network_structure():
    """
    打印网络结构信息
    """
    print("=" * 100)
    print(" " * 35 + "ReverberationNet 网状结构")
    print("=" * 100)
    
    try:
        from ReverbNet import ReverberationNet
        model = ReverberationNet(d=64, num_instruments=7)
        structure_info = model.get_network_structure()
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"总参数数量: {total_params:,}")
        print(f"角色模块数量: {len(structure_info['roles'])}")
        print(f"融合处理器数量: {len(structure_info['processors'])}")
        print(f"总连接数: {structure_info['total_connections']}")
        print()
        
        print("🎭 角色模块连接映射:")
        print("-" * 50)
        for i, role_name in enumerate(structure_info['roles'], 1):
            connected_processor = structure_info['role_to_processor'][role_name]
            print(f"{i:2d}. {role_name:<12} → {connected_processor}")
        
        print("\n🎵 融合处理器连接映射:")
        print("-" * 50)
        for processor_name, connected_roles in structure_info['processor_to_roles'].items():
            print(f"{processor_name:<12} ← {', '.join(connected_roles)}")
        
        print("\n🔗 网状连接特点:")
        print("• 每个角色通过门控机制选择连接到一个融合处理器")
        print("• 每个融合处理器接收多个角色的输入并融合处理")
        print("• 每个融合处理器输出反馈到3个角色")
        print("• 形成角色↔融合处理器的多层网状结构")
        print("• 所有信息最终汇聚到Argallia指挥层")
        print("=" * 100)
        
    except ImportError as e:
        print(f"导入ReverbNet失败: {e}")
        print("请确保ReverbNet.py文件存在且可正常导入")

def create_data_flow_diagram():
    """
    创建数据流程图
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # 标题
    ax.text(8, 9.5, 'ReverberationNet 数据流程图', fontsize=18, 
            ha='center', fontweight='bold', color='darkblue')
    
    # 定义各层的框
    boxes_info = [
        {"text": "输入 X\n(B, L, d)", "pos": (2, 8), "color": "lightblue"},
        {"text": "19个角色模块\n并行处理", "pos": (2, 6.5), "color": "lightgreen"},
        {"text": "门控选择\n连接融合处理器", "pos": (2, 5), "color": "yellow"},
        {"text": "7个融合处理器\n融合处理", "pos": (8, 6.5), "color": "orange"},
        {"text": "反馈分发\n3个角色/融合处理器", "pos": (8, 5), "color": "lightpink"},
        {"text": "角色接收\n融合处理器反馈", "pos": (14, 6.5), "color": "lightcyan"},
        {"text": "Argallia汇总\n全局注意力", "pos": (8, 3), "color": "gold"},
        {"text": "最终输出\n标量值", "pos": (8, 1.5), "color": "lightcoral"}
    ]
    
    # 绘制框
    for box in boxes_info:
        rect = FancyBboxPatch((box["pos"][0]-1, box["pos"][1]-0.4), 2, 0.8,
                             boxstyle="round,pad=0.1", facecolor=box["color"],
                             edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(box["pos"][0], box["pos"][1], box["text"], 
                fontsize=10, ha='center', va='center', fontweight='bold')
    
    # 绘制箭头连接
    arrows = [
        ((2, 7.6), (2, 6.9)),    # 输入→角色
        ((2, 6.1), (2, 5.4)),    # 角色→门控
        ((3, 5), (7, 6.5)),      # 门控→融合处理器
        ((8, 6.1), (8, 5.4)),    # 融合处理器→反馈
        ((9, 5), (13, 6.5)),     # 反馈→角色
        ((2, 4.6), (7, 3.4)),    # 角色→Argallia
        ((8, 4.6), (8, 3.4)),    # 融合处理器→Argallia
        ((14, 6.1), (9, 3.4)),   # 反馈角色→Argallia
        ((8, 2.6), (8, 1.9)),    # Argallia→输出
    ]
    
    for start, end in arrows:
        arrow = patches.FancyArrowPatch(start, end, arrowstyle='->', 
                                       mutation_scale=15, color='blue', linewidth=2)
        ax.add_patch(arrow)
    
    # 添加说明文字
    explanations = [
        {"text": "分发", "pos": (2, 7.2), "size": 8},
        {"text": "门控", "pos": (2, 5.7), "size": 8},
        {"text": "选择", "pos": (5, 5.8), "size": 8},
        {"text": "反馈", "pos": (8, 5.7), "size": 8},
        {"text": "融合", "pos": (11, 5.8), "size": 8},
        {"text": "汇聚", "pos": (5, 3.8), "size": 8},
        {"text": "输出", "pos": (8, 2.2), "size": 8}
    ]
    
    for exp in explanations:
        ax.text(exp["pos"][0], exp["pos"][1], exp["text"], 
                fontsize=exp["size"], ha='center', color='red', fontweight='bold')
    
    # 网状连接示意
    ax.text(8, 0.5, '🔗 网状特点：角色↔融合处理器双向连接 + 全局Argallia汇聚', 
            fontsize=12, ha='center', fontweight='bold', style='italic')
    
    plt.tight_layout()
    plt.savefig('reverbnet_dataflow.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ 数据流图已保存到: reverbnet_dataflow.png")
    
    return fig

if __name__ == "__main__":
    print("🎼 ReverberationNet 网状架构可视化工具")
    print("=" * 60)
    
    # 打印网络结构
    print_network_structure()
    
    print("\n📊 开始生成可视化图表...")
    
    # 主架构图
    fig1 = draw_mesh_architecture()
    
    # 数据流图
    fig2 = create_data_flow_diagram()
    
    # 显示图形
    print("\n✅ 所有图表生成完成！")
    print("📁 生成的文件:")
    print("   • reverbnet_architecture.png - 网状架构图")
    print("   • reverbnet_dataflow.png - 数据流程图")
    
    try:
        plt.show()
    except:
        print("注意：无法显示图形界面，但图片已保存到文件") 