import torch
from torch.utils.data import DataLoader, TensorDataset
import random
from ReverbNet import ReverberationNet

# 设置随机种子以保证可重复性
torch.manual_seed(42)
random.seed(42)

# --- 超参数 ---
batch_size = 32
input_dim = 64  # 输入特征维度 d
seq_len = 10    # 序列长度 L
num_epochs = 50  # 减少epoch数量以便快速测试
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"使用设备: {device}")

# --- 数据生成 ---
# 随机生成输入 (B, L, d) 和目标 (B,)
num_samples = 1000
X = torch.randn(num_samples, seq_len, input_dim)
y = torch.rand(num_samples)  # 假设是回归任务，输出标量

# 创建 DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 初始化模型和优化器 ---
print("初始化ReverberationNet网状模型...")
print("  • 19个角色模块 - 特化特征提取")
print("  • 7个数据处理器 - 数据聚合路由传输")
print("  • 1个Argallia指挥层 - 全局汇聚")
model = ReverberationNet(d=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 显示模型信息
total_params = sum(p.numel() for p in model.parameters())
structure_info = model.get_network_structure()
print(f"模型总参数数量: {total_params:,}")
print(f"角色模块数量: {len(structure_info['roles'])}")
print(f"数据处理器数量: {len(structure_info['processors'])}")
print(f"总连接数: {structure_info['total_connections']}")

# --- 训练模型 ---
print("\n开始网状训练...")
print("训练流程: 角色→门控选择→数据处理器→反馈→Argallia→输出")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    total_mse_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, kl_loss, gate_probs = model(inputs)

        # 回归任务使用 MSE Loss
        mse_loss = torch.nn.MSELoss()(outputs.squeeze(), targets.float())
        loss = mse_loss + kl_loss * 1e-4  # 调整KL权重，适配数据处理器

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse_loss += mse_loss.item()
        total_kl_loss += kl_loss.item()

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse_loss / len(dataloader)
    avg_kl = total_kl_loss / len(dataloader)
    
    # 每个epoch都打印详细信息
    print(f"Epoch {epoch+1:3d}/{num_epochs} | "
          f"Total Loss: {avg_loss:.6f} | "
          f"MSE Loss: {avg_mse:.6f} | "
          f"KL Loss: {avg_kl:.2f} | "
          f"Gate Probs: {len(gate_probs):2d}")

# --- 测试模型 ---
print("\n开始网状评估...")
model.eval()
total_mse = 0
with torch.no_grad():
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _, gate_probs = model(inputs)
        total_mse += torch.nn.MSELoss()(outputs.squeeze(), targets.float()).item()

avg_mse = total_mse / len(dataloader)
print(f"最终评估 MSE Loss: {avg_mse:.6f}")

# --- 显示门控概率信息 ---
print("\n角色→数据处理器门控概率分析:")
with torch.no_grad():
    sample_input = X[:1].to(device)  # 取一个样本
    _, _, gate_probs = model(sample_input)
    
    module_names = [
        "Eileen", "Pluto", "Organ", "Harp", "WolfHour", "Viola", 
        "Philip", "Cello", "CircusMaster", "Bremen", "Zaixian", 
        "Elena", "Greta", "Clarinet", "Horn", "Tuba", "Trombone", 
        "Violin1", "Violin2"
    ]
    
    print("各角色模块门控概率 (连接到数据处理器的强度):")
    for i, (name, prob) in enumerate(zip(module_names, gate_probs)):
        print(f"{i+1:2d}. {name:12s}: {prob.item():.4f}")

# --- 显示数据处理器连接信息 ---
print("\n数据处理器连接映射:")
role_to_processor = structure_info['role_to_processor']
processor_to_roles = structure_info['processor_to_roles']

processor_names = {
    'data_aggregator': '数据聚合器',
    'stream_processor': '流处理器', 
    'message_router': '消息路由器',
    'bandwidth_manager': '带宽管理器',
    'protocol_converter': '协议转换器',
    'cache_manager': '缓存管理器',
    'sync_coordinator': '同步协调器'
}

for processor_key, cn_name in processor_names.items():
    connected_roles = processor_to_roles.get(processor_key, [])
    print(f"📧 {processor_key:18s} ({cn_name})")
    print(f"   ← 连接角色: {', '.join(connected_roles)}")

print("\n🎯 网状连接特点:")
print("   • 每个角色通过门控机制选择数据处理器")
print("   • 数据处理器专注于数据聚合、路由、传输")
print("   • 形成角色↔数据处理器的网状信息流")
print("   • 所有信息最终汇聚到Argallia指挥层")

print("\n✅ 训练完成！数据处理器成功优化信息传输。")