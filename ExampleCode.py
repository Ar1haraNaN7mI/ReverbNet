import torch
from torch.utils.data import DataLoader, TensorDataset
import random

# 设置随机种子以保证可重复性
torch.manual_seed(42)
random.seed(42)

# --- 超参数 ---
batch_size = 32
input_dim = 64  # 输入特征维度 d
seq_len = 10    # 序列长度 L
max_layers = 12 # 最大网络层数（未使用但保留）
num_epochs = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 数据生成 ---
# 随机生成输入 (B, L, d) 和目标 (B,)
num_samples = 1000
X = torch.randn(num_samples, seq_len, input_dim)
y = torch.rand(num_samples)  # 假设是回归任务，输出标量

# 创建 DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 初始化模型和优化器 ---
model = ReverberationNet(d=input_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 训练模型 ---
print("开始训练...")
model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs, kl_loss, gate_probs = model(inputs)

        # 回归任务使用 MSE Loss
        mse_loss = torch.nn.MSELoss()(outputs.squeeze(), targets.float())
        loss = mse_loss + kl_loss * 1e-3  # KL 散度加权项

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

# --- 测试模型 ---
print("\n开始评估...")
model.eval()
total_mse = 0
with torch.no_grad():
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs, _, _ = model(inputs)
        total_mse += torch.nn.MSELoss()(outputs.squeeze(), targets.float()).item()

avg_mse = total_mse / len(dataloader)
print(f"Evaluation MSE Loss: {avg_mse:.6f}")