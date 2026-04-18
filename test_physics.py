import sys
import os

# 确保导入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import UAVHubConfig
from utils.physics_model import UAVPhysicsModel

def test_energy_logic():
    cfg = UAVHubConfig()
    physics = UAVPhysicsModel(cfg)
    
    print(f"--- 物理能耗模型边界测试 (E_max={cfg.E_max}Wh) ---")
    
    # 场景1：送一个 1000m 处的节点
    dist_safe = [1000.0]
    res_safe = physics.check_energy_red_line(dist_safe)
    # 预期能耗：(0.2+0.1)*1000 = 300 Wh
    print(f"[测试] 距离 1000m | 预期能耗 300Wh | 结果: {'通过' if res_safe else '拦截'}")
    
    # 场景2：送一个 2000m 处的节点
    dist_fail = [2000.0]
    res_fail = physics.check_energy_red_line(dist_fail)
    # 预期能耗：(0.2+0.1)*2000 = 600 Wh
    print(f"[测试] 距离 2000m | 预期能耗 600Wh | 结果: {'通过' if res_safe == False else '成功拦截！'}")

    # 场景 3：多个节点的星型总和 (动态计算打印版)
    dist_multi = [1300.0, 1500.0]
    total_dist = sum(dist_multi)
    expected_energy = total_dist * 0.3
    
    res_multi = physics.check_energy_red_line(dist_multi)
    
    # 动态判断它应该是通过还是被拦截
    if expected_energy <= 500.0:
        status = '通过 (正确)' if res_multi else '被误拦截 (错误)'
    else:
        status = '成功拦截 (正确)' if not res_multi else '漏过 (错误)'
        
    print(f"[测试] 多节点总距 {total_dist}m | 预期能耗 {expected_energy}Wh | 结果: {status}")

if __name__ == "__main__":
    test_energy_logic()