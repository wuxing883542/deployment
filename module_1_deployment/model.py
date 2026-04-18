import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseGCNLayer(nn.Module):
    """
    标准的图卷积层 (Graph Convolutional Layer)
    用于处理密集的图结构数据，实现 AXW 的信息聚合
    """
    def __init__(self, in_dim, out_dim):
        super(DenseGCNLayer, self).__init__()
        # 申请内存，建造一个负责翻译(维度转换)的 Linear 机器，这就是需要学习的参数 W
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, A):
        """
        前向传播
        X: 节点特征矩阵 (N, in_dim)
        A: 归一化后的邻接矩阵 (N, N)
        """
        # 第一步：XW (对自身特征进行降维/升维打击)
        support = self.linear(X)
        
        # 第二步：A * (XW) (根据邻居的权重，把大家的特征加权合并)
        out = torch.matmul(A, support)
        
        # 第三步：ReLU 铁面无私的安检门，过滤掉没用的负面情绪
        return F.relu(out)


class UAVPolicyNetwork(nn.Module):
    def __init__(self, num_nodes=20, node_feature_dim=3, hidden_dim=128):
        super(UAVPolicyNetwork, self).__init__()
        self.N = num_nodes
        
        # --- 1. 图特征提取器 (The Eyes) ---
        # 两层 GCN，让每个节点不仅知道自己，还能打听清楚邻居的八卦
        self.gcn1 = DenseGCNLayer(node_feature_dim, hidden_dim)
        self.gcn2 = DenseGCNLayer(hidden_dim, hidden_dim)
        
        # --- 2. Critic 价值评估头 (The Brain) ---
        # 评估局势：从 128 维的大局观，压缩成 64 个指标，最后拍板给出一个标量 V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) 
        )
        
        # --- 3. Actor 策略头 (The Mouth) ---
        # 动作空间平铺：5种操作 * N个操作节点 * N个目标节点 = 5 * N * N 个离散动作
        self.action_dim = 5 * self.N * self.N
        
        # 战术决策：从 128 维大局观，发散成 256 种战术意图，最后给 2000 个按钮挨个打分 (Logits)
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim) 
        )

    def _build_graph_features(self, state, C_matrix, D_array, f_array):
        """
        数据预处理车间：
        把物理环境的字典数据，翻译成神经网络能看懂的张量 (Tensor)
        """
        # 【核心探针】获取当前神经网络所在的厂房(CPU还是GPU)，确保数据不送错地方
        device = next(self.parameters()).device
        
        # 1. 构造节点特征矩阵 X (N x 3)
        # 特征包含: [当前是否为枢纽(0/1), 该点需求量, 该点建站成本]
        Y_kk = torch.FloatTensor(state['Y_kk']).to(device)
        D = torch.FloatTensor(D_array).to(device)
        f = torch.FloatTensor(f_array).to(device)
        
        # 归一化处理 (极其重要，把一切压缩到 0~1，防止神经网络梯度爆炸)
        D_norm = D / D.max()
        f_norm = f / f.max()
        
        # 像三明治一样叠起来 Shape: (20, 3)
        X = torch.stack([Y_kk, D_norm, f_norm], dim=1) 
        
        # 2. 构造邻接矩阵 A (N x N)
        C_tensor = torch.FloatTensor(C_matrix).to(device)
        tau = C_tensor.mean() # 动态温度系数
        
        # 距离越近，关系越强。使用高斯核函数将距离转化为 0~1 的亲密权重
        A = torch.exp(-C_tensor / tau) 
        
        # 加上自环：大声告诉 GCN，听八卦的同时别忘了自己是谁
        A = A + torch.eye(self.N, device=device)
        
        # 行归一化：加权平均，保证信息平滑传递
        A = A / A.sum(dim=1, keepdim=True)
        
        return X, A

   
    def forward(self, state, C_matrix, D_array, f_array):
        """
        【Exp_017 纯净版】前向传播流水线：看图 -> 思考 -> 输出纯净的动作打分和局势估值
        """
        # 1. 数据预处理
        X, A = self._build_graph_features(state, C_matrix, D_array, f_array)
        
        # 2. 图卷积特征提取
        h1 = self.gcn1(X, A)
        node_embeddings = self.gcn2(h1, A) # Shape: (20, 128)
        
        # 3. 读出机制 (Readout)：高管会议总结
        graph_embedding = torch.mean(node_embeddings, dim=0) # Shape: (128,)
        
        # 4. 兵分两路：Critic 价值评估 和 Actor 动作偏好打分
        state_value = self.critic_head(graph_embedding)
        action_logits = self.actor_head(graph_embedding)
        
        # 🚨 [史诗级减负]：删除了内部 Mask 逻辑！
        # 把最原始的冲动（Raw Logits）交出去，外部的环境和 train_ppo 会负责拦截！
        return action_logits, state_value

# =====================================================================
# 本地防刺客神智检查 (Sanity Check)
# =====================================================================
if __name__ == "__main__":
    import sys
    import os
    
    # 【核心防御】先把项目根目录强行加入系统视野
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    # 视野开阔了，再进行导包绝对不会报错
    from config import UAVHubConfig
    from env_deployment import UAVHubEnv
    
    print("[System] 正在加载 PyTorch 和环境，请耐心等待几秒钟...")
    
    cfg = UAVHubConfig()
    env = UAVHubEnv(cfg)
    state = env.reset()
    
    # 探测设备并把大脑扔上去
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] 检测到可用设备: {device.type.upper()}")
    
    brain = UAVPolicyNetwork(num_nodes=cfg.N).to(device)
    
    # 模拟一次前向传播思考
    logits, value = brain(state, env.C, env.D, env.f)
    
    print(f"[Model Test] 动作得分向量维度: {logits.shape} (应为 {5*20*20})")
    print(f"[Model Test] 当前状态评估价值: {value.item():.4f}")
    print(f"[Model Test] 网络初始化完全成功，设备隔离完美运行！")