import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 融合处理器层定义 ---
class FusionProcessor(nn.Module):
    """
    融合处理器：接收来自角色的信息，进行融合处理，输出到连接的角色
    每个融合处理器连接3个角色（简化但保持功能性）
    """
    def __init__(self, d, processor_name):
        super().__init__()
        self.d = d
        self.processor_name = processor_name
        
        # 简化的输入融合层
        self.input_fusion = nn.Linear(d, d)
        
        # 融合处理器特有的处理层（简化）
        self.fusion_processor = nn.Sequential(
            nn.Linear(d, d*2),
            nn.GELU(),
            nn.Linear(d*2, d)
        )
        
        # 输出分发层（分发到3个连接的角色）
        self.output_distributors = nn.ModuleList([
            nn.Linear(d, d) for _ in range(3)
        ])
        
        # 变分编码
        self.W_mu = nn.Linear(d, d)
        self.W_logvar = nn.Linear(d, d)
        
    def forward(self, role_inputs):
        """
        role_inputs: list of tensors from connected roles
        """
        # 简化的多角色输入融合
        if len(role_inputs) > 1:
            # 直接平均融合，然后通过线性层
            avg_input = torch.stack(role_inputs, dim=1).mean(dim=1)  # (B, L, d)
        else:
            avg_input = role_inputs[0]
        
        # 输入融合
        fused_output = self.input_fusion(avg_input)
        
        # 融合处理器特有处理
        processed = self.fusion_processor(fused_output.mean(dim=1, keepdim=True))  # (B, 1, d)
        
        # 变分编码
        mu = self.W_mu(processed)
        logvar = self.W_logvar(processed)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 分发到3个输出
        outputs = []
        for distributor in self.output_distributors:
            output = distributor(z)  # (B, 1, d)
            outputs.append(output.expand(-1, avg_input.size(1), -1))  # expand to original length
        
        return outputs, mu, logvar

# --- 角色模块基类 ---
class RoleModule(nn.Module):
    """
    角色模块基类
    """
    def __init__(self, d, role_name):
        super().__init__()
        self.d = d
        self.role_name = role_name
        
        # 门控选择层（选择连接到哪个乐器）
        self.instrument_gate = nn.Linear(d, 1)  # 输出选择概率
        
        # 角色特有的处理层 - 子类必须实现
        self.role_processor = self._build_role_processor()
        
        # 变分编码
        self.W_mu = nn.Linear(d, d)
        self.W_logvar = nn.Linear(d, d)
        
        # 层归一化
        self.ln = nn.LayerNorm(d)
        
    def _build_role_processor(self):
        """子类必须实现这个方法来定义角色特有的处理逻辑"""
        raise NotImplementedError("每个角色必须实现自己的特化处理逻辑")
    
    def forward(self, x, instrument_inputs=None):
        """
        x: 主输入
        instrument_inputs: 来自乐器层的输入列表
        """
        # 角色特有处理
        role_output = self.role_processor(x)
        
        # 如果有来自乐器的输入，进行融合
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            # 简单的加权融合
            instrument_sum = sum(instrument_inputs)
            # 确保维度匹配
            if instrument_sum.size(1) != role_output.size(1):
                # 如果序列长度不匹配，使用自适应平均池化调整
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), role_output.size(1)
                ).transpose(1,2)
            role_output = role_output + instrument_sum
        
        # 变分编码
        mu = self.W_mu(role_output)
        logvar = self.W_logvar(role_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 门控选择
        gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
        
        # 残差连接和层归一化
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

# --- 具体角色模块实现 - 每个角色都有专有特化 ---
class Eileen(RoleModule):
    """艾琳 - 高频调节与卷积注意力专家"""
    def __init__(self, d):
        super().__init__(d, "Eileen")
    
    def _build_role_processor(self):
        # 高频滤波 + 多头注意力
        return nn.ModuleDict({
            'conv': nn.Conv1d(self.d, self.d, kernel_size=3, padding=1),
            'attention': nn.MultiheadAttention(self.d, 4, batch_first=True),
            'ffn': nn.Linear(self.d, self.d)
        })
    
    def forward(self, x, instrument_inputs=None):
        # 高频滤波处理
        x_conv = self.role_processor['conv'](x.transpose(1,2)).transpose(1,2)
        x_conv = F.gelu(x_conv)
        
        # 多头注意力
        attn_output, _ = self.role_processor['attention'](x_conv, x_conv, x_conv)
        
        # 前馈网络
        role_output = self.role_processor['ffn'](attn_output)
        
        # 融合乐器输入
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != role_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), role_output.size(1)
                ).transpose(1,2)
            role_output = role_output + instrument_sum
        
        # 变分编码和门控
        mu = self.W_mu(role_output)
        logvar = self.W_logvar(role_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class Pluto(RoleModule):
    """普鲁托 - 深度非线性变换专家"""
    def __init__(self, d):
        super().__init__(d, "Pluto")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d*4, self.d*2),
            nn.GELU(),
            nn.Linear(self.d*2, self.d)
        )

class Organ(RoleModule):
    """管风琴 - 多管道并行处理专家"""
    def __init__(self, d):
        super().__init__(d, "Organ")
        # 多管道并行处理
        self.pipes = nn.ModuleList([nn.Linear(self.d, self.d) for _ in range(4)])
        self.fusion = nn.Linear(self.d*4, self.d)
    
    def _build_role_processor(self):
        return nn.Identity()  # 使用自定义forward
    
    def forward(self, x, instrument_inputs=None):
        # 多管道处理
        pipe_outputs = [F.gelu(pipe(x)) for pipe in self.pipes]
        concatenated = torch.cat(pipe_outputs, dim=-1)
        processed = self.fusion(concatenated)
        
        # 融合乐器输入
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != processed.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), processed.size(1)
                ).transpose(1,2)
            processed = processed + instrument_sum
        
        # 变分编码和门控
        mu = self.W_mu(processed)
        logvar = self.W_logvar(processed)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(processed.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class Harp(RoleModule):
    """竖琴 - LSTM序列建模专家"""
    def __init__(self, d):
        super().__init__(d, "Harp")
    
    def _build_role_processor(self):
        return nn.LSTM(self.d, self.d, batch_first=True)
    
    def forward(self, x, instrument_inputs=None):
        lstm_output, _ = self.role_processor(x)
        
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != lstm_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), lstm_output.size(1)
                ).transpose(1,2)
            lstm_output = lstm_output + instrument_sum
        
        mu = self.W_mu(lstm_output)
        logvar = self.W_logvar(lstm_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(lstm_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class WolfHour(RoleModule):
    """狼之时刻 - 双向GRU时域专家"""
    def __init__(self, d):
        super().__init__(d, "WolfHour")
    
    def _build_role_processor(self):
        return nn.GRU(self.d, self.d, batch_first=True, bidirectional=True)
    
    def forward(self, x, instrument_inputs=None):
        gru_output, _ = self.role_processor(x)
        # 双向GRU输出需要处理
        gru_output = gru_output[:, :, :self.d] + gru_output[:, :, self.d:]
        
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != gru_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), gru_output.size(1)
                ).transpose(1,2)
            gru_output = gru_output + instrument_sum
        
        mu = self.W_mu(gru_output)
        logvar = self.W_logvar(gru_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(gru_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class Viola(RoleModule):
    """中提琴 - 中频谐波分析专家"""
    def __init__(self, d):
        super().__init__(d, "Viola")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.Tanh(),  # 中提琴使用tanh激活
            nn.Linear(self.d, self.d),
            nn.LayerNorm(self.d)
        )

class Philip(RoleModule):
    """菲利普 - 结构调和与正则化专家"""
    def __init__(self, d):
        super().__init__(d, "Philip")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d//2),
            nn.ReLU(),
            nn.Linear(self.d//2, self.d),
            nn.Dropout(0.2),  # 正则化
            nn.Linear(self.d, self.d)
        )

class Cello(RoleModule):
    """大提琴 - 低频共鸣与深度特征专家"""
    def __init__(self, d):
        super().__init__(d, "Cello")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d*2),
            nn.LeakyReLU(0.2),  # 低频特性
            nn.Linear(self.d*2, self.d),
            nn.BatchNorm1d(self.d)
        )
    
    def forward(self, x, instrument_inputs=None):
        # 特殊处理BatchNorm
        role_output = self.role_processor[0](x)
        role_output = self.role_processor[1](role_output)
        role_output = self.role_processor[2](role_output)
        
        # BatchNorm需要转置
        B, L, D = role_output.shape
        role_output = self.role_processor[3](role_output.view(B*L, D)).view(B, L, D)
        
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != role_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), role_output.size(1)
                ).transpose(1,2)
            role_output = role_output + instrument_sum
        
        mu = self.W_mu(role_output)
        logvar = self.W_logvar(role_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class CircusMaster(RoleModule):
    """马戏团长 - 噪声控制与注意力调节专家"""
    def __init__(self, d):
        super().__init__(d, "CircusMaster")
    
    def _build_role_processor(self):
        return nn.ModuleDict({
            'noise_gate': nn.Linear(self.d, self.d),
            'attention': nn.MultiheadAttention(self.d, 2, batch_first=True),
            'control': nn.Linear(self.d, self.d)
        })
    
    def forward(self, x, instrument_inputs=None):
        # 噪声门控
        noise_controlled = torch.sigmoid(self.role_processor['noise_gate'](x)) * x
        
        # 注意力调节
        attn_output, _ = self.role_processor['attention'](noise_controlled, noise_controlled, noise_controlled)
        
        # 控制层
        role_output = self.role_processor['control'](attn_output)
        
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != role_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), role_output.size(1)
                ).transpose(1,2)
            role_output = role_output + instrument_sum
        
        mu = self.W_mu(role_output)
        logvar = self.W_logvar(role_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class Bremen(RoleModule):
    """不莱梅乐队 - 多声道融合专家"""
    def __init__(self, d):
        super().__init__(d, "Bremen")
        # 多声道处理
        self.channels = nn.ModuleList([nn.Linear(self.d, self.d) for _ in range(3)])
        
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d*3, self.d*2),
            nn.GELU(),
            nn.Linear(self.d*2, self.d)
        )
    
    def forward(self, x, instrument_inputs=None):
        # 多声道处理
        channel_outputs = [channel(x) for channel in self.channels]
        multi_channel = torch.cat(channel_outputs, dim=-1)
        role_output = self.role_processor(multi_channel)
        
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != role_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), role_output.size(1)
                ).transpose(1,2)
            role_output = role_output + instrument_sum
        
        mu = self.W_mu(role_output)
        logvar = self.W_logvar(role_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class Zaixian(RoleModule):
    """在宪 - 附旋律协同控制专家"""
    def __init__(self, d):
        super().__init__(d, "Zaixian")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.Sigmoid(),  # 协同控制使用sigmoid
            nn.Linear(self.d, self.d*2),
            nn.GELU(),
            nn.Linear(self.d*2, self.d)
        )

class Elena(RoleModule):
    """伊莲娜 - 主旋律建模专家"""
    def __init__(self, d):
        super().__init__(d, "Elena")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d*3),  # 主旋律需要更大容量
            nn.GELU(),
            nn.Linear(self.d*3, self.d*2),
            nn.GELU(),
            nn.Linear(self.d*2, self.d),
            nn.LayerNorm(self.d)
        )

class Greta(RoleModule):
    """格蕾塔 - 节奏结构编码专家"""
    def __init__(self, d):
        super().__init__(d, "Greta")
    
    def _build_role_processor(self):
        return nn.ModuleDict({
            'rhythm_conv': nn.Conv1d(self.d, self.d, kernel_size=5, padding=2),
            'tempo_linear': nn.Linear(self.d, self.d),
            'beat_gate': nn.Linear(self.d, self.d)
        })
    
    def forward(self, x, instrument_inputs=None):
        # 节奏卷积
        rhythm_output = self.role_processor['rhythm_conv'](x.transpose(1,2)).transpose(1,2)
        rhythm_output = F.gelu(rhythm_output)
        
        # 节拍处理
        tempo_output = self.role_processor['tempo_linear'](rhythm_output)
        beat_gate = torch.sigmoid(self.role_processor['beat_gate'](x))
        
        role_output = tempo_output * beat_gate
        
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != role_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), role_output.size(1)
                ).transpose(1,2)
            role_output = role_output + instrument_sum
        
        mu = self.W_mu(role_output)
        logvar = self.W_logvar(role_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class Clarinet(RoleModule):
    """单簧管 - 音色处理与频域变换专家"""
    def __init__(self, d):
        super().__init__(d, "Clarinet")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.Softplus(),  # 音色特性
            nn.Linear(self.d, self.d//2),
            nn.ELU(),
            nn.Linear(self.d//2, self.d)
        )

class Horn(RoleModule):
    """圆号 - 音域扩展与动态范围专家"""
    def __init__(self, d):
        super().__init__(d, "Horn")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d*4),  # 音域扩展
            nn.SiLU(),  # 平滑的音域过渡 (替代Mish)
            nn.Linear(self.d*4, self.d*2),
            nn.SiLU(),
            nn.Linear(self.d*2, self.d)
        )

class Tuba(RoleModule):
    """大号 - 低频增强与重低音专家"""
    def __init__(self, d):
        super().__init__(d, "Tuba")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.LeakyReLU(0.3),  # 低频特性
            nn.Linear(self.d, self.d*2),
            nn.LeakyReLU(0.3),
            nn.Linear(self.d*2, self.d),
            nn.LayerNorm(self.d)  # 改为LayerNorm
        )

class Trombone(RoleModule):
    """长号 - 滑音处理与连续变换专家"""
    def __init__(self, d):
        super().__init__(d, "Trombone")
    
    def _build_role_processor(self):
        return nn.ModuleDict({
            'slide_conv': nn.Conv1d(self.d, self.d, kernel_size=7, padding=3),
            'glide_linear': nn.Linear(self.d, self.d),
            'smooth_gate': nn.Linear(self.d, 1)
        })
    
    def forward(self, x, instrument_inputs=None):
        # 滑音卷积
        slide_output = self.role_processor['slide_conv'](x.transpose(1,2)).transpose(1,2)
        slide_output = F.gelu(slide_output)
        
        # 滑动变换
        glide_output = self.role_processor['glide_linear'](slide_output)
        
        # 平滑门控
        smooth_weight = torch.sigmoid(self.role_processor['smooth_gate'](x))
        role_output = glide_output * smooth_weight
        
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != role_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), role_output.size(1)
                ).transpose(1,2)
            role_output = role_output + instrument_sum
        
        mu = self.W_mu(role_output)
        logvar = self.W_logvar(role_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class Violin1(RoleModule):
    """第一小提琴 - 主声部领奏专家"""
    def __init__(self, d):
        super().__init__(d, "Violin1")
    
    def _build_role_processor(self):
        return nn.ModuleDict({
            'leader_attention': nn.MultiheadAttention(self.d, 8, batch_first=True),
            'vibrato_linear': nn.Linear(self.d, self.d),
            'expression_gate': nn.Linear(self.d, self.d)
        })
    
    def forward(self, x, instrument_inputs=None):
        # 主声部注意力
        leader_output, _ = self.role_processor['leader_attention'](x, x, x)
        
        # 颤音处理
        vibrato_output = self.role_processor['vibrato_linear'](leader_output)
        vibrato_output = F.gelu(vibrato_output)
        
        # 表现力门控
        expression_gate = torch.sigmoid(self.role_processor['expression_gate'](x))
        role_output = vibrato_output * expression_gate
        
        if instrument_inputs is not None and len(instrument_inputs) > 0:
            instrument_sum = sum(instrument_inputs)
            if instrument_sum.size(1) != role_output.size(1):
                instrument_sum = F.adaptive_avg_pool1d(
                    instrument_sum.transpose(1,2), role_output.size(1)
                ).transpose(1,2)
            role_output = role_output + instrument_sum
        
        mu = self.W_mu(role_output)
        logvar = self.W_logvar(role_output)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        gate_prob = torch.sigmoid(self.instrument_gate(role_output.mean(dim=1)))
        output = self.ln(x + z)
        
        return output, mu, logvar, gate_prob.squeeze(-1)

class Violin2(RoleModule):
    """第二小提琴 - 副声部和声专家"""
    def __init__(self, d):
        super().__init__(d, "Violin2")
    
    def _build_role_processor(self):
        return nn.Sequential(
            nn.Linear(self.d, self.d*2),
            nn.GELU(),
            nn.Linear(self.d*2, self.d),
            nn.Dropout(0.1),
            nn.Linear(self.d, self.d),
            nn.LayerNorm(self.d)
        )

# --- Argallia 指挥层 ---
class Argallia(nn.Module):
    """
    阿尔加利亚指挥层：汇总所有角色和融合处理器的输出
    """
    def __init__(self, d, num_roles, num_processors):
        super().__init__()
        self.d = d
        self.num_roles = num_roles
        self.num_processors = num_processors
        
        # 多头注意力汇总
        self.global_attention = nn.MultiheadAttention(d, 8, batch_first=True)
        
        # 最终处理层
        self.final_processor = nn.Sequential(
            nn.Linear(d, d//2),
            nn.LayerNorm(d//2),
            nn.GELU(),
            nn.Linear(d//2, 1)
        )
        
    def forward(self, role_outputs, processor_outputs):
        """
        role_outputs: list of role outputs
        processor_outputs: list of processor outputs
        """
        # 汇总所有输出
        all_outputs = role_outputs + processor_outputs
        
        # 确保所有张量有相同的序列长度
        target_length = role_outputs[0].size(1)  # 使用第一个角色输出的序列长度作为目标
        
        normalized_outputs = []
        for output in all_outputs:
            if output.size(1) != target_length:
                # 使用自适应平均池化调整序列长度
                output_normalized = F.adaptive_avg_pool1d(
                    output.transpose(1,2), target_length
                ).transpose(1,2)
            else:
                output_normalized = output
            normalized_outputs.append(output_normalized)
        
        # 堆叠所有输出
        stacked_outputs = torch.stack(normalized_outputs, dim=1)  # (B, num_sources, L, d)
        B, num_sources, L, d = stacked_outputs.shape
        stacked_outputs = stacked_outputs.view(B, num_sources * L, d)
        
        # 全局注意力
        attended, _ = self.global_attention(stacked_outputs, stacked_outputs, stacked_outputs)
        
        # 全局平均池化
        global_feature = attended.mean(dim=1)  # (B, d)
        
        # 最终输出
        output = self.final_processor(global_feature)
        
        return output.squeeze(-1)  # (B,)

# --- 主网络 ReverberationNet ---
class ReverberationNet(nn.Module):
    def __init__(self, d, num_instruments=7):
        super().__init__()
        self.d = d
        self.num_instruments = num_instruments
        
        # 19个角色模块
        self.roles = nn.ModuleDict({
            'eileen': Eileen(d),
            'pluto': Pluto(d),
            'organ': Organ(d),
            'harp': Harp(d),
            'wolfhour': WolfHour(d),
            'viola': Viola(d),
            'philip': Philip(d),
            'cello': Cello(d),
            'circusmaster': CircusMaster(d),
            'bremen': Bremen(d),
            'zaixian': Zaixian(d),
            'elena': Elena(d),
            'greta': Greta(d),
            'clarinet': Clarinet(d),
            'horn': Horn(d),
            'tuba': Tuba(d),
            'trombone': Trombone(d),
            'violin1': Violin1(d),
            'violin2': Violin2(d)
        })
        
        # 融合处理器层
        processor_names = ['Harmony', 'Rhythm', 'Melody', 'Texture', 'Dynamics', 'Timbre', 'Structure']
        self.processors = nn.ModuleDict({
            name.lower(): FusionProcessor(d, name) for name in processor_names[:num_instruments]
        })
        
        # 角色到融合处理器的连接映射（每个角色选择一个处理器）
        self.role_to_processor = self._create_role_processor_mapping()
        
        # 融合处理器到角色的连接映射（每个处理器连接3个角色）
        self.processor_to_roles = self._create_processor_role_mapping()
        
        # Argallia指挥层
        self.argallia = Argallia(d, len(self.roles), len(self.processors))
        
    def _create_role_processor_mapping(self):
        """创建角色到处理器的映射"""
        role_names = list(self.roles.keys())
        processor_names = list(self.processors.keys())
        
        mapping = {}
        for i, role_name in enumerate(role_names):
            # 每个角色连接到一个处理器（循环分配）
            processor_idx = i % len(processor_names)
            mapping[role_name] = processor_names[processor_idx]
        
        return mapping
    
    def _create_processor_role_mapping(self):
        """创建处理器到角色的映射（每个处理器连接3个角色）"""
        processor_names = list(self.processors.keys())
        role_names = list(self.roles.keys())
        
        mapping = {}
        roles_per_processor = 3
        
        for i, processor_name in enumerate(processor_names):
            # 每个处理器连接3个角色
            connected_roles = []
            for j in range(roles_per_processor):
                role_idx = (i * roles_per_processor + j) % len(role_names)
                connected_roles.append(role_names[role_idx])
            mapping[processor_name] = connected_roles
        
        return mapping
    
    def forward(self, x):
        """
        网状前向传播
        """
        batch_size, seq_len, _ = x.shape
        kl_loss = 0
        gate_probs = []
        
        # 第一层：所有角色并行处理输入
        role_outputs = {}
        role_mus = {}
        role_logvars = {}
        
        for role_name, role_module in self.roles.items():
            output, mu, logvar, gate_prob = role_module(x)
            role_outputs[role_name] = output
            role_mus[role_name] = mu
            role_logvars[role_name] = logvar
            gate_probs.append(gate_prob)
            kl_loss += self.kl_divergence(mu, logvar)
        
        # 第二层：处理器层处理
        processor_outputs = {}
        for processor_name, processor in self.processors.items():
            # 收集连接到这个处理器的角色输出
            connected_role_names = self.processor_to_roles[processor_name]
            role_inputs = [role_outputs[role_name] for role_name in connected_role_names]
            
            # 处理器处理
            inst_outputs, mu, logvar = processor(role_inputs)
            processor_outputs[processor_name] = inst_outputs
            kl_loss += self.kl_divergence(mu, logvar)
        
        # 第三层：角色接收来自处理器的反馈
        final_role_outputs = {}
        for role_name, role_module in self.roles.items():
            # 找到这个角色连接的处理器
            connected_processor = self.role_to_processor[role_name]
            
            # 找到这个角色在处理器输出中的位置
            processor_role_list = self.processor_to_roles[connected_processor]
            if role_name in processor_role_list:
                role_idx = processor_role_list.index(role_name)
                processor_input = [processor_outputs[connected_processor][role_idx]]
            else:
                processor_input = []
            
            # 角色再次处理（接收处理器反馈）
            output, mu, logvar, gate_prob = role_module(role_outputs[role_name], processor_input)
            final_role_outputs[role_name] = output
            gate_probs.append(gate_prob)
            kl_loss += self.kl_divergence(mu, logvar)
        
        # 最终层：Argallia汇总
        all_role_outputs = list(final_role_outputs.values())
        all_processor_outputs = []
        for inst_outputs in processor_outputs.values():
            all_processor_outputs.extend(inst_outputs)
        
        final_output = self.argallia(all_role_outputs, all_processor_outputs)
        
        return final_output, kl_loss, gate_probs
    
    def kl_divergence(self, mu, logvar):
        """计算KL散度"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def get_network_structure(self):
        """返回网络连接结构信息"""
        structure_info = {
            'roles': list(self.roles.keys()),
            'processors': list(self.processors.keys()),
            'role_to_processor': self.role_to_processor,
            'processor_to_roles': self.processor_to_roles,
            'total_connections': len(self.role_to_processor) + sum(len(roles) for roles in self.processor_to_roles.values())
        }
        return structure_info
    
    def train_model(self, dataloader, optimizer, epoch_num=10, device='cuda'):
        self.to(device)
        self.train()
        for epoch in range(epoch_num):
            total_loss = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs, kl_loss, gate_probs = self.forward(inputs)

                # 回归任务使用MSE Loss
                mse_loss = nn.MSELoss()(outputs.squeeze(), targets.float())
                loss = mse_loss + kl_loss * 1e-4  # 调整KL权重

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

