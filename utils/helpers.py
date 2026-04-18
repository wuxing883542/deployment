import random
import numpy as np
import torch

def set_global_seed(seed: int = 42):
    """
    固定全局随机种子，确保实验的绝对可复现性
    """
    # 1. 固定 Python 内置随机模块
    random.seed(seed)
    
    # 2. 固定 Numpy 随机种子
    np.random.seed(seed)
    
    # 3. 固定 PyTorch 随机种子 (CPU)
    torch.manual_seed(seed)
    
    # 4. 如果后续用到 GPU，固定 CUDA 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多 GPU 时使用
        # 为了保证卷积等操作的确定性 (虽然我们目前只用线性层和图卷积)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"[Info] Global seed set to {seed}. Reproducibility locked.")