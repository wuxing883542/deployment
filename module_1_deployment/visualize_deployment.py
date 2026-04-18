import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 【核心寻址】确保脚本能找到根目录的 config 和 data
# ==========================================
current_module_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_module_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.helpers import set_global_seed
from config import UAVHubConfig
from env_deployment import UAVHubEnv
from model import UAVPolicyNetwork

def decode_action(action_idx, num_nodes=20):
    """
    【动作解码器】
    将 1D 索引还原为 (动作类型, 节点索引, 目标索引)
    """
    action_type = action_idx // (num_nodes * num_nodes)
    node_idx = (action_idx % (num_nodes * num_nodes)) // num_nodes
    target_idx = action_idx % num_nodes
    return int(action_type), int(node_idx), int(target_idx)

def run_inference(policy, env, cfg, device, num_trials=50):
    """
    【推理模式 - 终极照相机 + 灵魂骰子 + 暴力采样版】
    - num_trials: 探索次数。利用算力换取极致的全局最优解。
    """
    global_best_cost = float('inf')
    global_best_Y_kk = None
    global_best_Y_ik = None
    global_best_step = 0
    global_best_trial = 0
    
    policy.eval()
    
    print("\n" + "="*50)
    print(f"🚀 开始执行暴力抽样推理 (开启照相模式，共探索 {num_trials} 次)...")
    print("="*50)
    
    action_names = {
        0: "Allocate (划拨)", 1: "Alternate(换帅)", 
        2: "Locate   (抢点)", 3: "AddHub   (建站)", 
        4: "RemoveHub(拆站)"
    }
    
    with torch.no_grad(): 
        for trial in range(1, num_trials + 1):
            state = env.reset()
            done = False
            step_count = 0
            banned_actions = set() # 动态黑名单，防止死循环
            
            if trial % 10 == 1 or trial == num_trials:
                print(f"--- 🌌 正在进行第 {trial} ~ {min(trial+9, num_trials)} 次世界线探索 ---")
            
            while not done and step_count < cfg.max_steps:
                # 1. 大脑思考 (输出 Raw Logits)
                logits, _ = policy(state, env.C, env.D, env.f)
                logits = logits.view(-1)
                
                # ==========================================
                # 🛡️ 【修复点】绝对防御：改向环境索要掩码 (Exp_017 协议)
                # ==========================================
                # 直接调用环境推演出来的合法动作面具 (5 x N x N)
                mask_np = env.get_action_mask() 
                
                # 展平并转换为 Tensor
                mask_tensor = torch.tensor(mask_np.flatten(), dtype=torch.bool).to(device)
                
                # 应用掩码：屏蔽物理与逻辑非法动作
                logits[~mask_tensor] = -1e9
                # ==========================================
                
                # 2. 动态黑名单拦截
                for ba in banned_actions:
                    logits[ba] = -1e9
                    
                if (logits == -1e9).all():
                    break
                
                # 3. 灵魂骰子采样
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action_idx = dist.sample().item()
                
                # 解析动作
                act_type, n_idx, t_idx = decode_action(action_idx, cfg.N)
                
                previous_cost = env.current_cost
                
                # 执行动作
                state, reward, done, _ = env.step(act_type, n_idx, t_idx)
                
                # 4. 裁判与记录阶段
                if env.current_cost == previous_cost:
                    banned_actions.add(action_idx)
                    continue  
                else:
                    # 照相机抓拍：只要比历史最低还低，立刻按下快门！
                    if env.current_cost < global_best_cost:
                        global_best_cost = env.current_cost
                        global_best_Y_kk = env.Y_kk.copy() 
                        global_best_Y_ik = env.Y_ik.copy()
                        global_best_step = step_count
                        global_best_trial = trial
                        
                        action_name = action_names.get(act_type, f"Type_{act_type}")
                        print(f"   🏆 [神图抓拍 - 第 {trial} 宇宙] 动作: {action_name} | 成本击穿至: {global_best_cost:.2f} 📉")
                    
                    banned_actions.clear()
                    step_count += 1
            
    print("\n" + "="*50)
    print(f"🎉 {num_trials} 次多重宇宙暴力探索圆满结束！")
    print(f"🥇 巅峰神图诞生于: 第 {global_best_trial} 次探索的第 {global_best_step} 步！")
    print(f"💰 最终决定成本: {global_best_cost:.2f}")
    print("="*50)
            
    return global_best_Y_kk, global_best_Y_ik, global_best_cost

def plot_topology(Y_kk, Y_ik, nodes_coords, cost, title, save_path, D=None, C=None, physics=None):
    """
    【绘图大师 - 终极富信息版】
    将 AI 的抽象决策转化为直观的地理拓扑图，并附带枢纽遥测数据！
    """
    plt.figure(figsize=(12, 9)) 
    N = len(Y_kk)
    
    # 1. 绘制归属连线 (灰虚线)
    for i in range(N):
        hub_idx = Y_ik[i]
        start_node = nodes_coords[i]
        end_hub = nodes_coords[hub_idx]
        plt.plot([start_node[0], end_hub[0]], [start_node[1], end_hub[1]], 
                 c='gray', linestyle='--', alpha=0.5, zorder=1)

    # 2. 绘制节点与枢纽看板
    for i in range(N):
        if Y_kk[i] == 1:
            # --- 🌟 枢纽点绘制 (红色星) ---
            plt.scatter(nodes_coords[i, 0], nodes_coords[i, 1], 
                        marker='*', s=300, c='red', 
                        label='Hub' if i == np.argmax(Y_kk) else "", 
                        edgecolors='black', zorder=3)
            
            # --- 📊 统计枢纽负载数据 ---
            if D is not None and C is not None and physics is not None:
                spoke_indices = np.where(Y_ik == i)[0]
                pure_spokes = spoke_indices[spoke_indices != i] 
                
                node_count = len(pure_spokes)
                total_weight = np.sum(D[pure_spokes])
                
                total_energy_cost = 0.0
                for spoke in pure_spokes:
                    dist = C[spoke, i]
                    # 复用物理引擎的经济计算
                    total_energy_cost += physics.calculate_economic_cost(dist, D[spoke])
                
                # --- 📝 绘制数据看板 ---
                info_text = f"Spokes: {node_count}\nDemand: {total_weight:.1f}\nCost: {total_energy_cost:.1f}"
                plt.text(nodes_coords[i, 0] + 15, nodes_coords[i, 1] + 15, 
                         info_text, fontsize=9, color='darkred', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.4'),
                         zorder=4)
        else:
            # --- 🔵 普通需求点绘制 (蓝色圆) ---
            plt.scatter(nodes_coords[i, 0], nodes_coords[i, 1], 
                        marker='o', s=80, c='blue', 
                        label='Node' if i == np.argmin(Y_kk) else "", 
                        alpha=0.8, zorder=2)

    # 3. 细节修饰
    plt.title(f"{title}\nTotal Cost: {cost:.2f}", fontsize=14, fontweight='bold')
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🖼️  图片已保存至: {save_path}")
    plt.close()

def main():
    set_global_seed(42)

    cfg = UAVHubConfig()
    env = UAVHubEnv(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载节点地理坐标 (对应 map_20n_seed42.csv 的 X, Y 列)
    csv_path = os.path.join(project_root, "data", "map_20n_seed42.csv")
    df = pd.read_csv(csv_path)
    nodes_coords = df[['X', 'Y']].values

    models_dir = os.path.join(current_module_dir, "models")
    data_dir = os.path.join(project_root, "data")
    
    # 待验证模型列表
    model_types = {
        "best_policy.pth": "Best Deployment Strategy (Lowest Cost)",
        "final_policy.pth": "Final Deployment Strategy (End of Training)"
    }

    for model_file, title in model_types.items():
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            print(f"⚠️  未找到模型文件: {model_file}，跳过生成。")
            continue
            
        print(f"\n🔍 正在加载策略权重: {model_file}...")
        
        policy = UAVPolicyNetwork(num_nodes=cfg.N).to(device)
        policy.load_state_dict(torch.load(model_path, map_location=device))
        
        # 执行推理，获取全局最优拓扑
        Y_kk, Y_ik, final_cost = run_inference(policy, env, cfg, device, num_trials=50)
        
        # 绘图存档
        save_filename = f"{model_file.split('.')[0]}_map.png"
        save_path = os.path.join(data_dir, save_filename)
        plot_topology(Y_kk, Y_ik, nodes_coords, final_cost, 
                      title, save_path, 
                      D=env.D, C=env.C, physics=env.physics)
        
    print("\n🎉 枢纽部署可视化流程全部验收完成！")

if __name__ == "__main__":
    main()