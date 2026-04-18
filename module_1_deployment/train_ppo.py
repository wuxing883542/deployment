import os
import sys
import copy

# ==========================================
# 【核心防御：跨目录寻址】
# 获取当前文件的绝对路径，并向上推一层找到项目根目录
# ==========================================
current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)

# 强行将项目根目录加入系统的环境变量视野中，确保能找到 config.py
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
# 【新增】TensorBoard 画笔工具
from torch.utils.tensorboard import SummaryWriter

# 视野开阔后，可以直接导入外层的 config
from config import UAVHubConfig
# 引入当前文件夹下的环境与模型
from env_deployment import UAVHubEnv
from model import UAVPolicyNetwork

# ==========================================
# 【新增】引入并激活全局随机种子，确保实验可复现！
# ==========================================
from utils.helpers import set_global_seed
set_global_seed(42)


def decode_action(action_idx, num_nodes=20):
    """
    【动作解码器】
    将 AI 输出的 1D 动作索引 (0~1999) 还原为环境可执行的 3D 物理动作
    """
    action_type = action_idx // (num_nodes * num_nodes)
    node_idx = (action_idx % (num_nodes * num_nodes)) // num_nodes
    target_idx = action_idx % num_nodes
    return int(action_type), int(node_idx), int(target_idx)

def train():
    print("=================================================")
    print("🚀 启动 [完全体 PPO + 绝对防御掩码] 无人机枢纽强化学习...")
    
    # ==========================================
    # 【智能算力调度】探测设备：优先 GPU (CUDA)，其次 CPU
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================")
    print(f"🖥️  当前训练设备: {device.type.upper()}")
    if device.type == 'cuda':
        print(f"🔥 显卡型号: {torch.cuda.get_device_name(0)}")
    print("=================================================")

    # ==========================================
    # 【工程化规范】创建当前模块下的专属产物文件夹
    # ==========================================
    models_dir = os.path.join(current_module_dir, "models")
    logs_dir = os.path.join(current_module_dir, "logs")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 【初始化画笔】准备向 logs 文件夹写入监控数据
    writer = SummaryWriter(log_dir=logs_dir)
    
    print(f"📁 模型将保存在: {models_dir}")
    print(f"📈 监控日志将保存在: {logs_dir} (请使用 TensorBoard 查看)")
    print("=================================================")

    # 1. 初始化配置与环境
    cfg = UAVHubConfig()
    env = UAVHubEnv(cfg)
    
    # 2. 初始化大脑 (Actor-Critic 双头网络) 并推送到指定设备
    policy = UAVPolicyNetwork(num_nodes=cfg.N).to(device)
    optimizer = Adam(policy.parameters(), lr=cfg.lr)

    # ==========================================
    # 【Exp_004 修正】记录历史最低成本，唯 Cost 论！
    # ==========================================
    best_cost = float('inf') 
    
    # 开始经历无数次的试错与进化...
    for episode in range(cfg.max_episodes):
        state = env.reset()
        done = False
        
        # 记录这一局的“通关录像” (Trajectory)
        states_list = []
        actions_list = []
        log_probs_list = []
        values_list = []
        rewards_list = []
        
        step_count = 0
        
        # --------------------------------------------------
        # 【阶段一：与环境交互，下棋积攒经验 (Rollout)】
        # --------------------------------------------------
        # 🛡️ 新增一个列表，专门记录每一步的“合法动作清单(Mask)”
        masks_list = [] 
        
        while not done and step_count < cfg.max_steps:
            # 必须使用深拷贝保存状态，防止环境在下一步覆盖当前状态导致复盘错误
            states_list.append(copy.deepcopy(state))
            
            # 1. 大脑思考，输出所有可能动作的原始倾向分 (Logits)
            # 这里的输入 C, D, f 依然是 CPU 上的 numpy，模型内部会处理并转移到 GPU
            logits, value = policy(state, env.C, env.D, env.f)
            
            # ==========================================
            # 🛡️ 绝对防御结界开启：拦截非法动作 (Exp_017 核心)
            # ==========================================
            # A. 向裁判(环境)索要当前状态的合法动作面具 (Shape: 5 x N x N)
            mask_np = env.get_action_mask() 
            
            # B. 将 3D 面具展平为 1D，对齐网络输出的 action_idx
            mask_flat = mask_np.flatten() 
            
            # C. 转换为 Tensor 并推送到 GPU
            mask_tensor = torch.tensor(mask_flat, dtype=torch.bool).to(device)
            masks_list.append(mask_tensor) # 极其关键：保存历史面具供反刍复盘使用！
            
            # D. 【核心神技】使用 PyTorch 的 masked_fill_ 
            # 把所有不合法 (False) 的动作概率强行压死为 负无穷 (-1e9)
            logits = logits.masked_fill(mask_tensor == False, -1e9)
            # ==========================================
            
            # 2. 掷骰子选动作 (此时非法动作的概率已被彻底清零，绝对不会选中)
            dist = Categorical(logits=logits)
            action_idx = dist.sample() 
            
            # 3. 翻译动作，让物理环境执行 (环境在 CPU 上，所以需要 .item() 转回标量)
            act_type, n_idx, t_idx = decode_action(action_idx.item(), cfg.N)
            next_state, reward, done, info = env.step(act_type, n_idx, t_idx)
            
            # 4. 记录当前步的数据
            actions_list.append(action_idx)
            log_probs_list.append(dist.log_prob(action_idx)) 
            values_list.append(value)                        
            rewards_list.append(reward)                      
            
            # 5. 状态推进
            state = next_state
            step_count += 1
            
        # --------------------------------------------------
        # 【阶段二：PPO 数据预处理与优势计算】
        # --------------------------------------------------
        total_reward = sum(rewards_list)
        
        # 计算未来的累积真实回报 (Discounted Returns)
        returns = []
        R = 0
        for r in reversed(rewards_list):
            R = r + cfg.gamma * R
            returns.insert(0, R)
        
        # 将数据转为 Tensor 并推送到设备，同时 detach() 切断历史数据的梯度
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        old_log_probs = torch.stack(log_probs_list).detach().to(device)
        old_values = torch.stack(values_list).squeeze().detach().to(device)
        
        # 【核心】优势 Advantage = 真实拿到的总收益 - Critic 之前预期的收益
        advantages = returns - old_values
        
        # 【工业级技巧】：优势归一化。确保训练不会因为单次奖励数值过大而梯度爆炸
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # --------------------------------------------------
        # 【阶段三：完全体 PPO 核心更新阶段 (反刍学习)】
        # --------------------------------------------------
        # 🛡️ 提前把刚才收集的所有面具打包成 Tensor 送到 GPU
        old_masks = torch.stack(masks_list).to(device)
        
        for _ in range(cfg.ppo_epochs):
            new_log_probs = []
            new_values = []
            entropies = []
            
            # 重新让网络评估一遍刚才经历过的所有状态 (此时网络参数已微调)
            for i in range(len(states_list)):
                logits, v = policy(states_list[i], env.C, env.D, env.f)
                
                # ==========================================
                # 🛡️ 同步结界：反刍复盘时，必须戴上【当时的】面具！
                # ==========================================
                logits = logits.masked_fill(old_masks[i] == False, -1e9)
                
                dist = Categorical(logits=logits)
                
                new_log_probs.append(dist.log_prob(actions_list[i]))
                new_values.append(v)
                entropies.append(dist.entropy()) # 记录信息熵，用于鼓励探索
                
            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.stack(new_values).squeeze()
            entropy = torch.stack(entropies).mean()
            
            # 1. 计算新旧策略的比率 Ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 2. PPO 灵魂：Clip 截断限制 (防止步子迈太大导致策略崩盘)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * advantages
            
            # Actor 损失：取最小值并取负（因为我们想最大化收益，而优化器是最小化 Loss）
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 3. Critic 损失：均方误差损失，让预测越来越准
            critic_loss = F.mse_loss(new_values, returns)
            
            # 4. 融合终极 Loss：策略损失 + 价值损失 - 熵奖励(增加多样性)
            loss = actor_loss + 0.5 * critic_loss - cfg.entropy_coef * entropy
            
            # 梯度更新三部曲
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # --------------------------------------------------
        # 【阶段四：写入 TensorBoard 监控大屏】
        # --------------------------------------------------
        writer.add_scalar("Train/1_Reward", total_reward, episode)
        writer.add_scalar("Train/2_Cost", env.current_cost, episode)
        writer.add_scalar("Loss/Critic_Loss", critic_loss.item(), episode)
        writer.add_scalar("Loss/Actor_Loss", actor_loss.item(), episode)
        writer.add_scalar("Metric/Entropy", entropy.item(), episode)

        # --------------------------------------------------
        # 【阶段五：打印日志与最优模型存档】
        # --------------------------------------------------
        # 1. 发现并保存历史最优模型 (已修正为判断当前成本是否低于历史最低成本)
        if env.current_cost < best_cost:
            best_cost = env.current_cost
            best_model_path = os.path.join(models_dir, "best_policy.pth")
            torch.save(policy.state_dict(), best_model_path)
            print(f"⭐ [新纪录] 局数: {episode+1:03d} | 累积奖励: {total_reward:.2f} | 突破最低成本: {best_cost:.2f} | 模型已保存!")
            
        # 2. 定期监控进度
        elif (episode + 1) % 10 == 0:
            print(f"🎮 局数: {episode+1:03d}/{cfg.max_episodes} | 步数: {step_count:02d} | 累积奖励: {total_reward:.2f} | 当前成本: {env.current_cost:.2f}")

    # 训练结束后，保存一个最终版模型，并收起画笔
    final_model_path = os.path.join(models_dir, "final_policy.pth")
    torch.save(policy.state_dict(), final_model_path)
    writer.close()
    
    print("=================================================")
    print(f"✅ PPO 训练圆满结束！")
    print(f"📍 最优模型: {best_model_path}")
    print(f"📍 最终模型: {final_model_path}")
    print("=================================================")

if __name__ == "__main__":
    train()