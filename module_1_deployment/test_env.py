import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UAVHubConfig
from env_deployment import UAVHubEnv

def run_environment_test():
    print("="*50)
    print("🚀 [启动] UAVHubEnv 强化学习环境单元测试")
    print("="*50)
    
    # 1. 初始化配置与环境
    cfg = UAVHubConfig()
    # 为了方便测试，我们把最大步数改小一点
    cfg.max_steps = 10 
    env = UAVHubEnv(cfg)
    
    # 2. 测试环境重置 (Reset)
    print("\n[测试 1] 执行 env.reset()...")
    state = env.reset()
    print(f"✅ Reset 成功！")
    print(f"📊 初始状态 (State) 包含: Y_kk 维度 {state['Y_kk'].shape}, Y_ik 维度 {state['Y_ik'].shape}")
    print(f"💰 初始基准成本 (Baseline Cost): {env.current_cost:.2f}")
    
    # 3. 暴力测试环境交互 (Step)
    print("\n[测试 2] 开始执行随机破坏性动作测试...")
    total_reward = 0
    
    for step in range(1, 15): # 故意跑 15 步，测试 max_steps = 10 的截断机制
        # 随机生成一个动作 (这里模拟 PPO 解码后的动作)
        action_type = np.random.randint(0, 3) # 0: Add, 1: Remove, 2: Allocate
        node_idx = np.random.randint(0, cfg.N)
        target_hub_idx = np.random.randint(0, cfg.N)
        
        print(f"\n--- 第 {step} 步 ---")
        action_names = ["AddHub(建站)", "RemoveHub(拆站)", "Allocate(改线)"]
        print(f"🤖 AI 尝试: {action_names[action_type]} | 操作节点: {node_idx} | 目标枢纽: {target_hub_idx}")
        
        # 核心交互！
        next_state, reward, done, info = env.step(action_type, node_idx, target_hub_idx)
        total_reward += reward
        
        print(f"📈 获得奖励 (Reward): {reward:.4f}")
        print(f"⚙️  内部状态 (Info): {info}")
        
        if done:
            print(f"\n🏁 [触发 Done] 环境在第 {step} 步正常结束！")
            print(f"🏆 累积总奖励: {total_reward:.4f}")
            print(f"💸 最终物流成本: {env.current_cost:.2f}")
            break

if __name__ == "__main__":
    run_environment_test()