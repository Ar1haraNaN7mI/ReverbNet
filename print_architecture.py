#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReverberationNet 网状架构概览打印工具
"""

def print_ascii_network_architecture():
    """
    在终端中打印ASCII艺术风格的网状架构图
    """
    print("=" * 120)
    print(" " * 40 + "🎼 ReverberationNet 网状架构图 🎼")
    print("=" * 120)
    print()
    
    # 输入部分
    print("📥 INPUT: (batch_size, sequence_length, feature_dim)")
    print("     ↓ (分发到所有角色)")
    print()
    
    # 第一层：角色模块
    print("🎭 第一层：19个角色模块并行处理")
    print("┌" + "─" * 116 + "┐")
    
    roles = [
        ("Eileen", "艾琳"), ("Pluto", "普鲁托"), ("Organ", "管风琴"), ("Harp", "竖琴"),
        ("WolfHour", "狼之时刻"), ("Viola", "中提琴"), ("Philip", "菲利普"), ("Cello", "大提琴"),
        ("CircusMaster", "马戏团长"), ("Bremen", "不莱梅"), ("Zaixian", "在宪"), ("Elena", "伊莲娜"),
        ("Greta", "格蕾塔"), ("Clarinet", "单簧管"), ("Horn", "圆号"), ("Tuba", "大号"),
        ("Trombone", "长号"), ("Violin1", "小提琴1"), ("Violin2", "小提琴2")
    ]
    
    # 分三行显示角色
    for i in range(0, len(roles), 7):
        row_roles = roles[i:i+7]
        role_line = "│ "
        for eng, cn in row_roles:
            role_line += f"{eng}({cn}) "
        role_line += " " * (114 - len(role_line)) + "│"
        print(role_line)
    
    print("└" + "─" * 116 + "┘")
    print("     ↓ (门控选择融合处理器)")
    print()
    
    # 融合处理器层
    print("🎵 第二层：7个融合处理器融合处理")
    print("┌" + "─" * 90 + "┐")
    processors = ["Harmony(和声)", "Rhythm(节奏)", "Melody(旋律)", "Texture(织体)", 
                  "Dynamics(力度)", "Timbre(音色)", "Structure(结构)"]
    
    proc_line = "│ "
    for proc in processors:
        proc_line += f"{proc} "
    proc_line += " " * (88 - len(proc_line)) + "│"
    print(proc_line)
    print("└" + "─" * 90 + "┘")
    print("     ↓ (反馈到角色)")
    print()
    
    # 第三层：角色接收反馈
    print("🔄 第三层：角色接收融合处理器反馈")
    print("┌" + "─" * 70 + "┐")
    print("│ 所有角色接收来自对应融合处理器的反馈信号，进行第二次处理        │")
    print("└" + "─" * 70 + "┘")
    print("     ↓ (汇聚)")
    print()
    
    # Argallia指挥层
    print("┌" + "─" * 60 + "┐")
    print("│ 🎯 Argallia (阿尔加利亚) - 指挥层 & 最终输出           │")
    print("│ 使用全局注意力机制汇聚所有角色和融合处理器的输出            │")
    print("└" + "─" * 60 + "┘")
    print("     ↓")
    print("📤 OUTPUT: (batch_size,) - 标量输出")
    
    print()
    print("🔗 网状连接机制:")
    print("   • 每个角色通过门控机制选择连接到一个融合处理器")
    print("   • 每个融合处理器接收多个角色输入并进行融合")
    print("   • 每个融合处理器输出反馈到3个角色")
    print("   • 形成复杂的角色↔融合处理器网状连接")
    
    print("=" * 120)

def print_connection_mapping():
    """
    打印详细的连接映射关系
    """
    print("\n🗺️ 网状连接映射详情")
    print("-" * 80)
    
    try:
        from ReverbNet import ReverberationNet
        
        model = ReverberationNet(d=64, num_instruments=7)
        structure_info = model.get_network_structure()
        
        print("📋 角色 → 融合处理器 映射（门控选择）:")
        print("-" * 50)
        for i, (role_name, processor_name) in enumerate(structure_info['role_to_processor'].items(), 1):
            print(f"{i:2d}. {role_name:<15} → {processor_name}")
        
        print("\n📋 融合处理器 ← 角色 映射（每个处理器连接3个角色）:")
        print("-" * 50)
        for processor_name, connected_roles in structure_info['processor_to_roles'].items():
            print(f"{processor_name:<12} ← [{', '.join(connected_roles)}]")
        
        print(f"\n📊 连接统计:")
        print(f"   • 总连接数: {structure_info['total_connections']}")
        print(f"   • 角色数: {len(structure_info['roles'])}")
        print(f"   • 融合处理器数: {len(structure_info['processors'])}")
        print(f"   • 平均每处理器连接角色数: {sum(len(roles) for roles in structure_info['processor_to_roles'].values()) / len(structure_info['processors']):.1f}")
        
    except ImportError:
        print("需要导入ReverbNet来显示详细连接映射")

def print_gate_mechanism():
    """
    打印门控机制说明
    """
    print("\n⚡ 门控机制说明")
    print("-" * 60)
    
    print("🚪 角色门控选择:")
    print("   • 每个角色模块包含门控层: Linear(d, 1) + Sigmoid")
    print("   • 输出概率值决定连接到哪个融合处理器的强度")
    print("   • 实现动态、自适应的处理器选择")
    
    print("\n🔀 融合处理器机制:")
    print("   • 简化融合来自多个角色的输入")
    print("   • 处理器特有的处理层进行信号变换")
    print("   • 变分编码确保信息的正则化")
    
    print("\n🔄 反馈分发:")
    print("   • 每个融合处理器有3个输出分发器")
    print("   • 将处理后的信息反馈给3个角色")
    print("   • 形成信息的循环和增强")

def print_model_statistics():
    """
    打印模型统计信息
    """
    print("\n📈 网状模型统计信息")
    print("-" * 50)
    
    try:
        import torch
        from ReverbNet import ReverberationNet
        
        model = ReverberationNet(d=64, num_instruments=7)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数数量:     {total_params:,}")
        print(f"可训练参数:     {trainable_params:,}")
        print(f"模型大小 (MB):  {total_params * 4 / 1024 / 1024:.2f}")
        
        # 分层统计
        role_params = sum(p.numel() for name, p in model.named_parameters() if 'roles' in name)
        processor_params = sum(p.numel() for name, p in model.named_parameters() if 'processors' in name)
        argallia_params = sum(p.numel() for name, p in model.named_parameters() if 'argallia' in name)
        
        print(f"角色层参数:     {role_params:,} ({role_params/total_params*100:.1f}%)")
        print(f"融合处理器层参数: {processor_params:,} ({processor_params/total_params*100:.1f}%)")
        print(f"指挥层参数:     {argallia_params:,} ({argallia_params/total_params*100:.1f}%)")
        
        # 测试前向传播
        x = torch.randn(1, 10, 64)
        with torch.no_grad():
            output, kl_loss, gate_probs = model(x)
        
        print(f"门控模块数量:   {len(gate_probs)}")
        print(f"平均门控概率:   {sum(gate_probs).item() / len(gate_probs):.4f}")
        print(f"KL散度:        {kl_loss.item():.6f}")
        
    except ImportError:
        print("需要导入torch和ReverbNet来显示详细统计")

def print_network_comparison():
    """
    对比新旧架构
    """
    print("\n🔄 架构演进对比")
    print("-" * 80)
    
    print("📜 旧架构（顺序连接）:")
    print("   • 19个模块按顺序连接")
    print("   • 特殊连接: Eileen → Argallia, Zaixian → Elena")
    print("   • 简单的线性信息流")
    print("   • 信息损失较大")
    
    print("\n🆕 新架构（网状连接）:")
    print("   • 19个角色 + 7个融合处理器层")
    print("   • 门控选择机制动态连接")
    print("   • 每个融合处理器连接3个角色")
    print("   • 多层网状信息交互")
    print("   • 更强的表达能力和信息保留")
    
    print("\n✅ 改进优势:")
    print("   • 增强了模型的非线性能力")
    print("   • 提供了更多的信息交互路径")
    print("   • 门控机制实现自适应连接")
    print("   • 反馈机制增强了信息流")

if __name__ == "__main__":
    print_ascii_network_architecture()
    print_connection_mapping()
    print_gate_mechanism()
    print_model_statistics()
    print_network_comparison()
    
    print("\n🎨 要查看图形化网状架构图，请运行:")
    print("   python visualize_architecture.py")
    print("\n📖 查看详细文档:")
    print("   README.md")
    print("\n🚀 运行训练示例:")
    print("   python ExampleCode.py") 