import gym
from gym import spaces
import numpy as np
import random
from rl_common.class_op import ClassOp, cal_scplx_matrix
from jy_exp.rl_common.data_loader import load_shared_data

class ClassIntegrationEnv(gym.Env):
    def __init__(self, classes, attributes, methods, method_counts, attr_counts, max_steps=5000, wM=0.5, wA=0.5, mu=0.0001, eta=0.0001, version='v1', NOF=[], GNN_class=None, seed=None):
        super(ClassIntegrationEnv, self).__init__()
        self.classes = classes
        self.attributes = attributes
        self.methods = methods
        self.method_counts = method_counts
        self.attr_counts = attr_counts
        self.num_classes = max(map(int, classes.keys()))
        print("num_classes:", self.num_classes)
        self.max_steps = max_steps
        self.wM = wM
        self.wA = wA
        print("wM:", wM)
        print("wA:", wA)
        self.mu = mu
        self.eta = eta
        self.version = version
        print("version="+version)
        self.influence = np.sum(attributes + methods, axis=0)
        self.complexity = np.sum(attributes + methods, axis=1)
        self.importance = (self.influence + self.complexity) / 2
        self.action_space = spaces.Discrete(self.num_classes)
        self.observation_space = spaces.MultiBinary(self.num_classes)
        self.state = None
        self.steps = 0
        self.best_reward = -np.inf
        self.current_sequence = []
        self.tested_classes = []        #已测试的类
        self.available_actions = list(range(self.num_classes))  # 可测类
        self.dependency_matrix = cal_scplx_matrix(self.methods, self.attributes, self.num_classes,)
        if self.version == 'v2':
            self.CBO = ClassOp.cal_CBO(self.methods, self.attributes, self.num_classes)
            self.NOF = NOF
            self.all_importance = ClassOp.calculate_class_importance(self.classes, self.NOF, self.CBO, self.dependency_matrix)
            # for class_idx, importance in self.all_importance.items():
            #     print(f"Class {class_idx}----{importance:.4f}")
            print("Class Importance Sequence:")
            print(" ".join(str(idx) for idx, _ in sorted(self.all_importance.items(), key=lambda x: x[1], reverse=True)))
        elif self.version == 'v2.1':
            self.version = 'v2'
            self.all_importance = GNN_class
            print("GNN Class Importance Sequence:")
            print(" ".join(str(idx) for idx, _ in sorted(self.all_importance.items(), key=lambda x: x[1], reverse=True)))
        self.seed_value = seed
        self._set_seed(self.seed_value)
        self.reset()
        #try
        self.gnn_class = GNN_class
    
    def _set_seed(self, seed=None):
        """设置环境随机种子"""
        np.random.seed(seed)
        random.seed(seed)
        self.seed_value = seed
        return [seed]


    def reset(self):
        if self.seed_value is not None:
            reset_seed = self.seed_value + self.steps
            np.random.seed(reset_seed)
            random.seed(reset_seed)
            
        self.state = np.zeros(self.num_classes, dtype=np.int32)
        self.steps = 0
        self.current_sequence = []
        self.tested_classes = []
        self.available_actions = list(range(self.num_classes))
        return self.state

    def step(self, action):
        # 更新状态和已测试类
        self.state[action] = 1
        if action in self.available_actions:
            self.available_actions.remove(action)

        if self.version == 'v1':
            done, reward = self.reward_function_v1(action)
        elif self.version == 'v2':
            done, reward = self.reward_function_v2(action)
        elif self.version == 'v2.2':
            done, reward = self.reward_function_v2_2(action)
        elif self.version == 'v3':
            done, reward = self.reward_function_v3(action)
        elif self.version == 'v4':
            done, reward = self.reward_function_v4(action)
        elif self.version == 'v5':
            done, reward = self.reward_function_v5(action)
        elif self.version == 'v6':
            done, reward = self.reward_function_v6(action)
        else:
            raise ValueError("Unknown reward function version")
        
        self.current_sequence.append(action)
        self.tested_classes.append(action)
        
        self.steps += 1
        # 判断是否结束
        if np.all(self.state == 1) or self.steps >= self.max_steps or not self.available_actions:
            done = True
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_sequence = self.current_sequence.copy()
        return self.state, reward, done, {}

    # 重要性
    def reward_function_v2(self, action):
        node_importance = self.all_importance.get(action+1, 0.0)
        done = False
        stub_complexity = 0
        c = 100   #100 1
        for i in range(self.num_classes):
            if self.state[i] == 0:
                Cplx = self.dependency_matrix[action][i]
                stub_complexity += Cplx
        reward = c*(1+node_importance - stub_complexity)
        return done, reward
    
    def reward_function_v2_2(self, action):
        MAX = 1000
        done = False
        if action in self.tested_classes:
            return True, -np.inf
        #self.state[action] = 1
        # 计算类的结构重要性（与之前相同）
        node_importance = (self.influence[action] + self.complexity[action]) / 2
        # 如果有GNN分数，则将其纳入考虑
        gnn_importance = self.gnn_class.get(action+1, 0.0)
        # 结合结构重要性和GNN重要性
        combined_importance = (node_importance + self.mu * gnn_importance)
        
        # 计算桩复杂度
        stub_complexity = 0
        for i in range(self.num_classes):
            if self.state[i] == 0:
                stub_complexity += (self.wM * (self.methods[action][i] ** 2) + self.wA * (self.attributes[action][i] ** 2))**0.5
                
        # 最终奖励：结合重要性减去桩复杂度
        reward = combined_importance - self.eta * stub_complexity
        
        # 确保奖励值在合理范围内
        reward = min(MAX, reward)
        return done, reward

    # 丁艳茹
    def reward_function_v6(self, action):
        MAX = 10
        c = 1
        done = False
        if action in self.tested_classes:
            return True, -1000
        stub_complexity = 0
        for i in range(self.num_classes):
            if self.state[i] == 0:
                cplx = (self.wM * (self.methods[action][i] ** 2) + self.wA * (self.attributes[action][i] ** 2))**0.5
                stub_complexity += cplx
        reward = c*(MAX - stub_complexity)
        return done, reward

    def render(self, mode='human'):
        pass

    def get_best_sequence(self):
        best_sequence_names = [self.classes[i] for i in self.best_sequence]
        return self.best_sequence, best_sequence_names, self.best_reward
