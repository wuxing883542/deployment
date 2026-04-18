import numpy as np
import copy
import sys
import os

# 确保导入路径正确，保证能找到根目录的 config 和 utils
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import UAVHubConfig
from utils.env_generate import load_instance
from utils.physics_model import UAVPhysicsModel 

class UAVHubEnv:
    def __init__(self, config: UAVHubConfig, instance_path: str = None):
        self.cfg = config
        self.physics = UAVPhysicsModel(config) # 接入物理法官
        
        # 1. 自动加载地图数据
        if instance_path is None:
            instance_path = os.path.join(BASE_DIR, 'data', f"map_{self.cfg.N}n_seed{self.cfg.seed}.pkl")
        instance_data = load_instance(instance_path)
        
        self.D = instance_data['D']
        self.C = instance_data['C']
        self.f = instance_data['f']
        self.lambda_cap = instance_data['lambda_cap']
        
        # 2. 初始化核心决策变量 (Y_kk: 建站决策, Y_ik: 分配决策)
        self.Y_kk = np.zeros(self.cfg.N, dtype=int)
        self.Y_ik = np.zeros(self.cfg.N, dtype=int)
        self.current_cost = float('inf')

    def reset(self):
        """回合开始：[Exp_006 大国重器] 全员建站 + 就地管辖"""
        
        # 1. 极其暴力的全员建站：20 个点全都是枢纽 (状态全为 1)
        self.Y_kk = np.ones(self.cfg.N, dtype=int)
        
        # 2. 极其安全的就地管辖：每个点只把货存给自己
        # np.arange 会生成 [0, 1, 2, ..., N-1]
        # 完美实现 Y_ik[i] = i (绝对不超电量，也绝对不触发库容)
        self.Y_ik = np.arange(self.cfg.N, dtype=int)
        
        # 3. 初始化成本结算并返回状态
        # 此时的开局成本会极其高昂 (20个建站费)，但安检门绝对全是绿灯！
        self.current_cost = self._calculate_total_cost(self.Y_kk, self.Y_ik)

        # ==========================================
        # 【Exp_009 核心】：真·跨地图通用自适应分母
        # 初始全员建站成本 / 节点数 = 单个枢纽的平均成本
        # 无论地图是 20 还是 20000，价值锚点永不漂移！
        # ==========================================
        self.reward_scale = self.current_cost / self.cfg.N
        
        return self._get_state()

    def get_action_mask(self):
        """
        【Exp_017 动作掩码生成器】：上帝视角的未来推演
        返回一个三维布尔矩阵 shape: (5, N, N) 
        代表 (action_type, node_idx, target_hub_idx) 的合法性。
        True(1) 代表合法可执行，False(0) 代表物理或逻辑违规绝对禁止！
        """
        # 初始化一个全 1 的掩码矩阵 (假设所有动作初始都是合法的)
        mask = np.ones((5, self.cfg.N, self.cfg.N), dtype=bool)
        
        # 遍历所有可能的动作组合进行未来推演
        for a_type in range(5):
            for n_idx in range(self.cfg.N):
                for t_idx in range(self.cfg.N):
                    
                    # 1. 快速过滤掉无意义的动作，节省算力
                    if a_type == 0 and self.Y_kk[n_idx] == 1: # 枢纽不能 Allocate
                        mask[a_type, n_idx, t_idx] = False
                        continue
                    if a_type == 3 and self.Y_kk[n_idx] == 1: # 已经是枢纽不能 AddHub
                        mask[a_type, n_idx, t_idx] = False
                        continue
                        
                    # ==========================================
                    # 🌟 转移过来的：防概率稀释魔法！
                    # ==========================================
                    # 算子 3(建站) 和 4(拆站) 根本不需要目标节点，强行锁定 t_idx = 0 才是合法的
                    # 否则会产生 N 个执行效果一模一样的多余动作，严重摊薄正确动作的概率
                    if (a_type == 3 or a_type == 4) and t_idx != 0:
                        mask[a_type, n_idx, t_idx] = False
                        continue
                        
                    # 2. 复制当前状态作为草稿纸
                    draft_Y_kk = self.Y_kk.copy()
                    draft_Y_ik = self.Y_ik.copy()
                    
                    # 3. 在草稿纸上试探性地执行这个动作
                    self._apply_operator(a_type, n_idx, t_idx, draft_Y_kk, draft_Y_ik)
                    
                    # 4. 如果执行完之后状态没变（说明是个无效的 ALNS 动作），也屏蔽掉
                    # 这完美代替了之前写在 model.py 里的几十行复杂 if-else 判断！
                    if np.array_equal(draft_Y_kk, self.Y_kk) and np.array_equal(draft_Y_ik, self.Y_ik):
                        mask[a_type, n_idx, t_idx] = False
                        continue
                        
                    # 5. 🚨 核心结界：将草稿纸送入物理安检门
                    if not self._is_state_valid(draft_Y_kk, draft_Y_ik):
                        # 如果未来会超载或断电，立刻把这个动作在掩码中拉黑！
                        mask[a_type, n_idx, t_idx] = False
                        
        # 【安全兜底】：如果某种极端情况下全盘皆墨（全是 False），为了防止网络崩溃
        # 我们强行打开一条活路：允许节点原地建站 (AddHub 绝对不会超载)
        # 🚨 极其关键的细节：为了配合上方的“防概率稀释”，这里兜底的 t_idx 必须设为 0
        if not mask.any():
            for i in range(self.cfg.N):
                mask[3, i, 0] = True
                
        return mask


    def _calculate_total_cost(self, Y_kk, Y_ik):
        """
        【Exp_017 动作掩码版】：纯净物理算盘
        总成本 = 运输费 + 建站费 (无需罚款，因为非法动作已被 Mask 在输出层拦截)
        """
        delivery_cost = 0.0
        # 建站成本
        building_cost = np.sum(self.f * Y_kk)
        
        # 正常计算经济成本 (纯物理耗能转换)
        for i in range(self.cfg.N):
            k = Y_ik[i]
            delivery_cost += self.physics.calculate_economic_cost(self.C[i, k], self.D[i])
                
        # 返回最纯粹的真实成本
        return delivery_cost + building_cost

    def _is_state_valid(self, prop_Y_kk, prop_Y_ik):
        """
        【Exp_017 物理结界】：纯净的物理安检门
        只校验能量和载重，没有任何经济逻辑，直接返回 True/False
        """
        for k in range(self.cfg.N):
            if prop_Y_kk[k] == 0: continue # 不是枢纽则跳过
            
            spoke_indices = np.where(prop_Y_ik == k)[0]
            pure_spokes = spoke_indices[spoke_indices != k]
            
            if len(pure_spokes) == 0: continue # 光杆司令枢纽绝对合法
            
            # 1. 🚨 单次起飞载重红线 (仅限外围派送需求)
            if np.sum(self.D[pure_spokes]) > self.cfg.Q:
                return False
                
            # 2. 🚨 能量保底红线
            if not self.physics.check_energy_red_line(self.C[pure_spokes, k]):
                return False
                
        return True # 全部通过，证明这个未来状态是合法的

    def _apply_operator(self, action_type, node_idx, target_hub_idx, prop_Y_kk, prop_Y_ik):
        """执行动作：在草稿纸上改变地图拓扑 (严格遵循 5 大核心算子官方定义)"""
        
        # ==========================================
        # 🟢 动作 0: Allocate 跨区划拨
        # 将边缘节点换绑到其他枢纽
        # ==========================================
        if action_type == 0: 
            if prop_Y_kk[node_idx] == 0 and prop_Y_kk[target_hub_idx] == 1:
                prop_Y_ik[node_idx] = target_hub_idx

        # ==========================================
        # 🟡 动作 1: Alternate 内部换帅
        # 同片区内更换枢纽位置，原片区小弟整体平移给新枢纽
        # ==========================================
        elif action_type == 1:
            if prop_Y_kk[node_idx] == 1 and prop_Y_kk[target_hub_idx] == 0:
                prop_Y_kk[node_idx] = 0        # 旧王退位
                prop_Y_kk[target_hub_idx] = 1  # 新王登基
                prop_Y_ik[target_hub_idx] = target_hub_idx
                
                # 同片区换帅：旧王的小弟直接整体过继给新王
                orphans = np.where(prop_Y_ik == node_idx)[0]
                prop_Y_ik[orphans] = target_hub_idx

        # ==========================================
        # 🔴 动作 2: Locate 跨区抢点 (🚨 高危算子：已打补丁)
        # 撤销当前枢纽，在别的片区新建。旧小弟必须就近重新找下家！
        # ==========================================
        elif action_type == 2:
            if prop_Y_kk[node_idx] == 1 and prop_Y_kk[target_hub_idx] == 0:
                # 1. 远方新王登基
                prop_Y_kk[target_hub_idx] = 1
                prop_Y_ik[target_hub_idx] = target_hub_idx
                
                # 2. 本地旧王被裁撤
                prop_Y_kk[node_idx] = 0 
                
                # 3. 旧王的小弟们（包括旧王自己）成难民了
                remaining = np.where(prop_Y_kk == 1)[0] # 包含刚刚登基的新王
                orphans = np.where(prop_Y_ik == node_idx)[0]
                
                # 🌟 难民自主寻家逻辑：不要强行分配给远方的新王，而是找全图最近的存活枢纽
                for orphan in orphans:
                    closest_to_orphan = remaining[np.argmin(self.C[orphan, remaining])]
                    prop_Y_ik[orphan] = closest_to_orphan

        # ==========================================
        # 🟢 动作 3: AddHub 增设枢纽
        # 应对局部载重或能量红线爆表的情况
        # ==========================================
        elif action_type == 3:
            if prop_Y_kk[node_idx] == 0:
                prop_Y_kk[node_idx] = 1        # 原地提拔为枢纽
                prop_Y_ik[node_idx] = node_idx # 独立门户

        # ==========================================
        # 🔴 动作 4: RemoveHub 裁撤枢纽 (🚨 高危算子：已打补丁)
        # 节约建站成本。同理，旧小弟必须就近重新找下家！
        # ==========================================
        elif action_type == 4: 
            if prop_Y_kk[node_idx] == 1 and np.sum(prop_Y_kk) > 1: # 保证地图至少留1个枢纽
                prop_Y_kk[node_idx] = 0 # 咔嚓！拆站
                
                remaining = np.where(prop_Y_kk == 1)[0] # 找出当前活着的枢纽
                orphans = np.where(prop_Y_ik == node_idx)[0] # 找出难民
                
                # 🌟 难民自主寻家逻辑
                for orphan in orphans:
                    closest_to_orphan = remaining[np.argmin(self.C[orphan, remaining])]
                    prop_Y_ik[orphan] = closest_to_orphan
    def step(self, action_type, node_idx, target_hub_idx=0):
        """【Exp_014 终极进化版 step】：一切皆是生意，彻底允许试错"""
        prop_Y_kk = copy.deepcopy(self.Y_kk)
        prop_Y_ik = copy.deepcopy(self.Y_ik)
        
        # 1. 执行动作改变草稿状态
        self._apply_operator(action_type, node_idx, target_hub_idx, prop_Y_kk, prop_Y_ik)
        
        # 🚨 [关键删除]：彻底废弃 _check_all_constraints 拦截代码！
        # 千万别再 return False 强制结束了！
        
        # 2. 算总账 (如果 AI 画了一条超长的线，new_cost 会因为加上了 5万罚款而瞬间飙升！)
        new_cost = self._calculate_total_cost(prop_Y_kk, prop_Y_ik)
        
        # 3. 计算价值差 (如果违规，delta 会变成巨大的负数)
        delta = self.current_cost - new_cost  
        
        # 4. 终极奖励公式：降本收益 + 步数税 (鞭策它去干活，绝不允许原地苟活！)
        reward = (delta / self.reward_scale) + self.cfg.step_penalty
        
        # 5. 🚨 环境无条件接受新状态！(哪怕被罚了款，状态也保留，逼它下一步自己用 Allocate 去修补烂摊子！)
        self.Y_kk, self.Y_ik, self.current_cost = prop_Y_kk, prop_Y_ik, new_cost 
        
        return self._get_state(), reward, False, {"msg": "State Updated with Penalty Check"}

    def _get_state(self):
        """返回当前环境观测值"""
        return {"Y_kk": self.Y_kk, "Y_ik": self.Y_ik}

if __name__ == "__main__":
    env = UAVHubEnv(UAVHubConfig())
    state = env.reset()
    print(f"[Success] 环境已就绪，物理模型已接入。初始全建站成本: {env.current_cost:.2f}")