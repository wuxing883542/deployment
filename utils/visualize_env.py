import os
import sys
import matplotlib.pyplot as plt

# 确保能导入根目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import UAVHubConfig
# 注意这里：我们把 export_to_csv_for_human 也导入进来了！
from utils.env_generate import generate_instance, save_instance, load_instance, export_to_csv_for_human

def plot_instance(instance, save_name=None):
    """可视化无人机物流节点分布、需求量和建站成本"""
    coords = instance['coords']
    demands = instance['D']
    costs = instance['f']
    N = len(coords)
    
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                          s=demands * 15, 
                          c=costs, 
                          cmap='YlOrRd', 
                          alpha=0.9, 
                          edgecolors='black')
    
    cbar = plt.colorbar(scatter)
    cbar.set_label('Hub Building Cost (f_k)', fontsize=12)
    
    # 【改动点】在这里把需求量具体数值加上
    for i in range(N):
        # f"{demands[i]:.1f}" 的意思是取需求量数值，保留1位小数
        label_text = f"ID:{i}\nD:{demands[i]:.1f}N"
        plt.text(coords[i, 0] + 12, coords[i, 1] + 12, label_text, fontsize=8, fontweight='bold')
        
    plt.title(f"UAV Logistic Map (N={N})\nNode Size: Demand | Node Color: Hub Cost", fontsize=14)
    plt.xlabel("X Coordinate (m)")
    plt.ylabel("Y Coordinate (m)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(-50, 1050)
    plt.ylim(-50, 1050)
    plt.tight_layout()

    if save_name:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        if not save_name.endswith(('.png', '.jpg', '.pdf')):
            save_name += '.png'
            
        save_path = os.path.join(data_dir, save_name)
        plt.savefig(save_path, dpi=300)
        print(f"[Success] 图像可视化已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    cfg = UAVHubConfig()
    
    # 1. 生成数据
    ins = generate_instance(cfg, seed=cfg.seed)
    
    # 定义统一的文件名前缀
    file_prefix = f"map_{cfg.N}n_seed{cfg.seed}"
    
    # 2. 保存为机器读取的 .pkl
    save_instance(ins, f"{file_prefix}.pkl")
    
    # 3. 【新增点】保存为人类查看的 .csv！
    export_to_csv_for_human(ins, f"{file_prefix}.csv")
    
    # 4. 可视化并保存图片
    plot_instance(ins, save_name=f"{file_prefix}.png")