import numpy as np
from scipy.spatial import distance_matrix
import os
import pickle
import csv
def generate_instance(cfg, seed=None):
    """生成一个无人机单向派送网络的算例"""
    if seed is not None:
        np.random.seed(seed)
        
    coords = np.random.uniform(0, cfg.map_size, size=(cfg.N, 2))
    C = distance_matrix(coords, coords)
    D = np.random.uniform(cfg.demand_min, cfg.demand_max, size=cfg.N)
    f = np.random.uniform(cfg.f_min, cfg.f_max, size=cfg.N)
    
    total_demand = np.sum(D)
    # 1. 汇总全图所有节点的需求总量 (假设约为 500N)
    lambda_cap = np.random.uniform(total_demand * 1.2, total_demand * 1.5, size=cfg.N)
    # 2. 为每个节点预设一个“如果它变身为枢纽”时的库容上限
    # 规则：每个枢纽只能承载全图总需求的 30% 到 50%
    # 意义：这意味着没有任何一个枢纽能搞“大一统”，
    # 物理上强制要求地图上必须存在多个枢纽来分担总货量。
    
    return {
        'coords': coords,
        'C': C,
        'D': D,
        'f': f,
        'lambda_cap': lambda_cap
    }

def save_instance(instance, filename="test_instance.pkl"):
    """将算例数据保存到 data 文件夹下"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    file_path = os.path.join(data_dir, filename)
    with open(file_path, 'wb') as file:
        pickle.dump(instance, file)
    print(f"[Success] 算例已成功保存至: {file_path}")
    return file_path

def load_instance(file_path):
    """从本地读取算例数据"""
    with open(file_path, 'rb') as file:
        instance = pickle.load(file)
    return instance


def export_to_csv_for_human(instance, filename="nodes_info.csv"):
    """
    将节点的属性(不包含庞大的距离矩阵)导出为人类可读的 CSV 表格
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    file_path = os.path.join(data_dir, filename)
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写表头
        writer.writerow(["Node_ID", "X", "Y", "Demand_D(N)", "Hub_Cost_f", "Capacity_Limit"])
        
        # 逐行写入 20 个节点的数据
        for i in range(len(instance['D'])):
            writer.writerow([
                i, 
                round(instance['coords'][i][0], 2),  # X 坐标
                round(instance['coords'][i][1], 2),  # Y 坐标
                round(instance['D'][i], 2),          # 需求量
                round(instance['f'][i], 2),          # 建站成本
                round(instance['lambda_cap'][i], 2)  # 枢纽容量上限
            ])
    print(f"[Export] 节点信息已导出为 CSV 用于人类查看: {file_path}")