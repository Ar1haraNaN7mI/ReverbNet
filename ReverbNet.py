import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Pluto角色模块 ---
class Pluto(nn.Module):
    """
    Pluto的机制：
    - 单头自注意力: attn(Q,K,V) = softmax(QK^T/sqrt(d))V
    - 变分编码: mu = W_mu(h), logvar = W_logvar(h)
    - 采样z = mu + eps * exp(0.5*logvar)
    - 门控: p = sigmoid(W_gate(h_mean))
    - 残差 + LayerNorm
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.W_mu = nn.Linear(d, d)
        self.W_logvar = nn.Linear(d, d)
        self.W_gate = nn.Linear(d, 1)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        Q = self.Wq(x)  # (B,1,d)
        K = self.Wk(x)  # (B,1,d)
        V = self.Wv(x)  # (B,1,d)
        # 计算注意力权重
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d)  # (B,1,1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn_weights, V)  # (B,1,d)
        # 变分编码参数
        mu = self.W_mu(h)
        logvar = self.W_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # 门控概率
        p_gate = torch.sigmoid(self.W_gate(h.mean(dim=1)))  # (B,1) -> (B,)
        # 输出带残差层归一化
        out = self.ln(x + z)
        return out, mu, logvar, p_gate.squeeze(-1)

# --- Eileen角色模块 ---
class Eileen(nn.Module):
    """
    Eileen机制：
    - 多头注意力(4头)，每头维度d/4
    - scaled dot-product attention per head，拼接
    - 变分编码用不同线性层
    - 激活用tanh，门控用两层MLP输出sigmoid
    """
    def __init__(self, d, n_heads=4):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_k = d // n_heads
        # Q,K,V映射
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        # 变分参数
        self.W_mu = nn.Linear(d, d)
        self.W_logvar = nn.Linear(d, d)
        # 门控MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, d//2),
            nn.ReLU(),
            nn.Linear(d//2,1),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        B, L, _ = x.size()
        Q = self.Wq(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)  # (B, heads, L, d_k)
        K = self.Wk(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        V = self.Wv(x).view(B, L, self.n_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, heads, L, d_k)
        out = out.transpose(1,2).contiguous().view(B, L, self.d)  # (B,L,d)
        mu = self.W_mu(out)
        logvar = self.W_logvar(out)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        out_activated = torch.tanh(z)
        p_gate = self.gate_mlp(out_activated.mean(dim=1))
        out_final = self.ln(x + out_activated)
        return out_final, mu, logvar, p_gate.squeeze(-1)

# --- Philip角色模块 ---
class Philip(nn.Module):
    """
    Philip机制：
    - 自注意力 + 前馈网络 (Feed Forward)
    - 变分编码直接在前馈输出层
    - 门控用softmax选取继续/停止概率
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d*2),
            nn.ReLU(),
            nn.Linear(d*2, d)
        )
        self.fc_mu = nn.Linear(d, d)
        self.fc_logvar = nn.Linear(d, d)
        self.gate_fc = nn.Linear(d, 2)  # 输出2维softmax (继续/停止)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn_weights, V)
        ffn_out = self.ffn(h)
        mu = self.fc_mu(ffn_out)
        logvar = self.fc_logvar(ffn_out)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        gate_logits = self.gate_fc(ffn_out.mean(dim=1))  # (B,2)
        p_gate = F.softmax(gate_logits, dim=-1)[:,0]  # 继续概率取第0个
        out = self.ln(x + z)
        return out, mu, logvar, p_gate

# --- BremenBand角色模块 ---
class BremenBand(nn.Module):
    """
    BremenBand机制：
    - 单头注意力但使用log(abs(x)+1e-5)激活变形
    - 变分编码采用不同激活
    - 门控直接线性层输出sigmoid
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.W_mu = nn.Linear(d, d)
        self.W_logvar = nn.Linear(d, d)
        self.W_gate = nn.Linear(d, 1)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn_weights, V)
        # 激活
        h = torch.log(torch.abs(h) + 1e-5)
        mu = self.W_mu(h)
        logvar = self.W_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        p_gate = torch.sigmoid(self.W_gate(h.mean(dim=1)))
        out = self.ln(x + torch.tanh(z))
        return out, mu, logvar, p_gate.squeeze(-1)

# --- Irene角色模块 ---
class Irene(nn.Module):
    """
    Irene机制：
    - 多头注意力，softmax加权后带残差跳跃
    - 变分编码后用ReLU激活
    - 门控采用两层带Dropout的MLP
    """
    def __init__(self, d, heads=2):
        super().__init__()
        self.d = d
        self.heads = heads
        self.d_k = d // heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.fc_mu = nn.Linear(d, d)
        self.fc_logvar = nn.Linear(d, d)
        self.gate = nn.Sequential(
            nn.Linear(d, d//2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(d//2, 1),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        B, L, _ = x.size()
        Q = self.Wq(x).view(B, L, self.heads, self.d_k).transpose(1,2)
        K = self.Wk(x).view(B, L, self.heads, self.d_k).transpose(1,2)
        V = self.Wv(x).view(B, L, self.heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(B,L,self.d)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = F.relu(mu + eps * std)
        p_gate = self.gate(z.mean(dim=1))
        out_final = self.ln(x + z)
        return out_final, mu, logvar, p_gate.squeeze(-1)

# --- Organ角色模块 ---
class Organ(nn.Module):
    """
    Organ机制：
    - 自注意力中引入位置编码（sinusoidal）
    - 变分编码，门控线性层
    """
    def __init__(self, d, max_len=50):
        super().__init__()
        self.d = d
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.W_mu = nn.Linear(d, d)
        self.W_logvar = nn.Linear(d, d)
        self.W_gate = nn.Linear(d, 1)
        self.ln = nn.LayerNorm(d)
        # 位置编码
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1,max_len,d)
    def forward(self, x):
        B, L, _ = x.size()
        Q = self.Wq(x + self.pe[:, :L])
        K = self.Wk(x + self.pe[:, :L])
        V = self.Wv(x + self.pe[:, :L])
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn_weights, V)
        mu = self.W_mu(h)
        logvar = self.W_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        p_gate = torch.sigmoid(self.W_gate(h.mean(dim=1)))
        out = self.ln(x + z)
        return out, mu, logvar, p_gate.squeeze(-1)

# --- Harp角色模块 ---
class Harp(nn.Module):
    """
    Harp机制：
    - 变分自编码器样式，隐空间用ReLU，门控用sigmoid
    - 注意力使用缩放因子不同
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.fc_mu = nn.Linear(d, d)
        self.fc_logvar = nn.Linear(d, d)
        self.fc_gate = nn.Linear(d, 1)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / (math.sqrt(self.d)*1.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn_weights, V)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = F.relu(mu + eps * std)
        p_gate = torch.sigmoid(self.fc_gate(h.mean(dim=1)))
        out = self.ln(x + z)
        return out, mu, logvar, p_gate.squeeze(-1)

# --- Aelwyn角色模块 ---
class Aelwyn(nn.Module):
    """
    Aelwyn机制：
    - 自注意力 + 残差块 + layer norm
    - 变分编码用不同权重
    - 门控用softmax二分类
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.fc_mu = nn.Linear(d, d)
        self.fc_logvar = nn.Linear(d, d)
        self.gate_fc = nn.Linear(d, 2)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        attn_scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h = torch.matmul(attn_weights, V)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        gate_logits = self.gate_fc(h.mean(dim=1))
        p_gate = F.softmax(gate_logits, dim=-1)[:,0]
        out = self.ln(x + z)
        return out, mu, logvar, p_gate

# --- OrganPipe角色模块 ---
class OrganPipe(nn.Module):
    """
    OrganPipe机制：
    - 使用两层自注意力
    - 变分编码后用elu激活
    - 门控用sigmoid
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.Wq1 = nn.Linear(d, d)
        self.Wk1 = nn.Linear(d, d)
        self.Wv1 = nn.Linear(d, d)
        self.Wq2 = nn.Linear(d, d)
        self.Wk2 = nn.Linear(d, d)
        self.Wv2 = nn.Linear(d, d)
        self.fc_mu = nn.Linear(d, d)
        self.fc_logvar = nn.Linear(d, d)
        self.fc_gate = nn.Linear(d, 1)
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        Q1 = self.Wq1(x)
        K1 = self.Wk1(x)
        V1 = self.Wv1(x)
        attn_scores1 = torch.matmul(Q1, K1.transpose(-2,-1)) / math.sqrt(self.d)
        attn_weights1 = F.softmax(attn_scores1, dim=-1)
        h1 = torch.matmul(attn_weights1, V1)
        Q2 = self.Wq2(h1)
        K2 = self.Wk2(h1)
        V2 = self.Wv2(h1)
        attn_scores2 = torch.matmul(Q2, K2.transpose(-2,-1)) / math.sqrt(self.d)
        attn_weights2 = F.softmax(attn_scores2, dim=-1)
        h2 = torch.matmul(attn_weights2, V2)
        mu = self.fc_mu(h2)
        logvar = self.fc_logvar(h2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = F.elu(mu + eps * std)
        p_gate = torch.sigmoid(self.fc_gate(h2.mean(dim=1)))
        out = self.ln(x + z)
        return out, mu, logvar, p_gate.squeeze(-1)

# --- Harpsichord角色模块 ---
class Harpsichord(nn.Module):
    """
    Harpsichord机制：
    - 多头自注意力(2头)
    - 变分编码后用selu激活
    - 门控用两层MLP + sigmoid
    """
    def __init__(self, d, heads=2):
        super().__init__()
        self.d = d
        self.heads = heads
        self.d_k = d // heads
        self.Wq = nn.Linear(d, d)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.fc_mu = nn.Linear(d, d)
        self.fc_logvar = nn.Linear(d, d)
        self.gate = nn.Sequential(
            nn.Linear(d, d//2),
            nn.SELU(),
            nn.Linear(d//2, 1),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(d)
    def forward(self, x):
        B, L, _ = x.size()
        Q = self.Wq(x).view(B, L, self.heads, self.d_k).transpose(1,2)
        K = self.Wk(x).view(B, L, self.heads, self.d_k).transpose(1,2)
        V = self.Wv(x).view(B, L, self.heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1,2).contiguous().view(B,L,self.d)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = F.selu(mu + eps * std)
        p_gate = self.gate(z.mean(dim=1))
        out_final = self.ln(x + z)
        return out_final, mu, logvar, p_gate.squeeze(-1)

# --- Algerian 指挥角色 ---
class Algerian(nn.Module):
    """
    阿尔加利亚层：
    - 全连接层融合所有特征为标量输出
    """
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, d//2),
            nn.ReLU(),
            nn.Linear(d//2, 1)
        )
    def forward(self, x):
        # x: (B, L, d)
        x_mean = x.mean(dim=1)  # (B, d)
        out = self.fc(x_mean)
        return out.squeeze(-1)  # (B,)

# --- Reverberation Net 主体 ---
class ReverberationNet(nn.Module):
    def __init__(self, d, max_layers=12):
        super().__init__()
        self.d = d
        self.max_layers = max_layers
        # 顺序堆叠角色
        self.pluto = Pluto(d)
        self.eileen = Eileen(d)
        self.philip = Philip(d)
        self.bremen = BremenBand(d)
        self.irene = Irene(d)
        self.organ = Organ(d)
        self.harp = Harp(d)
        self.aelwyn = Aelwyn(d)
        self.organpipe = OrganPipe(d)
        self.harpsichord = Harpsichord(d)
        # 可以按需增加模块，示范12个角色用10个了，可以补充两个或者复用部分
        self.algerian = Algerian(d)

    def forward(self, x):
        """
        x: (B,L,d) 输入特征序列
        返回: 标量输出 + 总损失(kl+reconstruction)
        """
        kl_loss = 0
        gate_probs = []
        mu_list = []
        logvar_list = []
        
        # 每个模块依次执行
        x, mu, logvar, p_gate = self.pluto(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.eileen(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.philip(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.bremen(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.irene(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.organ(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.harp(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.aelwyn(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.organpipe(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        x, mu, logvar, p_gate = self.harpsichord(x)
        kl_loss += self.kl_divergence(mu, logvar)
        gate_probs.append(p_gate)
        mu_list.append(mu)
        logvar_list.append(logvar)

        # 结束层输出
        out = self.algerian(x)
        return out, kl_loss, gate_probs
    def kl_divergence(self, mu, logvar):
        # 计算变分编码的KL散度
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    def train_model(self, dataloader, optimizer, epoch_num=10, device='cuda'):
        self.to(device)
        self.train()
        for epoch in range(epoch_num):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs, kl_loss, gate_probs = self.forward(inputs)

                # 假设是回归任务，使用MSE Loss
                mse_loss = nn.MSELoss()(outputs.squeeze(), targets.float())
                loss = mse_loss + kl_loss * 1e-3  # KL散度加权项，超参数可调

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epoch_num}, Loss: {avg_loss:.6f}")

    def evaluate_model(self, dataloader, device='cuda'):
        self.to(device)
        self.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, kl_loss, gate_probs = self.forward(inputs)
                mse_loss = nn.MSELoss()(outputs.squeeze(), targets.float())
                total_loss += mse_loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Evaluation MSE Loss: {avg_loss:.6f}")
        return avg_loss

