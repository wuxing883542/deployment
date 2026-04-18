import numpy as np
from dataclasses import dataclass

@dataclass
class UAVHubConfig:
    # --- 0. 全局控制 ---
    seed: int = 42              # 全局随机种子
    
    # --- 1. 集合定义与基础参数 ---
    N: int = 20                 # 缩小场景：20个节点
    
    # --- 2. 无人机物理红线参数 (全牛顿 N 版本) ---
    Q: float = 180.0            # 单次最大载重量 (N) 
    E_max: float = 1200.0        # 电池最大可用能量 (Wh)
    e_full: float = 0.2         # 满载状态单位距离能耗系数 (Wh/m)
    e_empty: float = 0.1        # 空载状态单位距离能耗系数 (Wh/m)
    
    # --- 3. 地图生成参数 ---
    map_size: float = 1000.0    # 地图边长 (m)，假设为正方形区域
    # --- 需求点参数 
    demand_min: float = 20.0    # 注意这里加上了 : float
    demand_max: float = 30.0    # 注意这里加上了 : float
    # --- 建站成本参数 
    f_min: float = 10000.0       # 注意这里加上了 : float
    f_max: float = 20000.0       # 注意这里加上了 : float


    # --- 4. 强化学习与 PPO 训练超参数 ---
    # 环境交互参数
    max_steps: int = 150             # 每局最大步数（防止AI陷入无意义的循环）
    step_penalty: float = -0.005     # 每走一步的固定时间惩罚（逼迫AI速战速决）
    death_penalty: float = -10.0    # 踩红线直接扣除巨额分数 (让AI痛入骨髓)
    embedding_dim: int = 128        # GCN输出特征维度

    # PPO 核心超参数
    lr: float = 3e-4                # 学习率 (Actor 和 Critic 共享)
    gamma: float = 0.99             # 奖励折扣率 (目光长远程度)
    clip_epsilon: float = 0.2       # PPO Clip 截断范围 (核心！限制每次进化的步子大小)
    max_episodes: int = 1200         # 总训练局数
    ppo_epochs: int = 4             # 经验复用：每局数据反复学习 4 次
    entropy_coef: float = 0.05      # 探索欲：越大越喜欢瞎试，一般设 0.01