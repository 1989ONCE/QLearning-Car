import numpy as np
import os, sys

# Q-Learning implementation
class QLearn() :
    def __init__(self, lrn_rate=0.1, gamma=0.9, epsilon=0.99, discount=0.9):
        self.lrn_rate = lrn_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = discount  # Decay rate for epsilon
        self.q_table = {}  # Q-table to store Q-values
        self.action_space = np.arange(-45, 46, 1)  # 動作空間：-45°到45°的範圍
        self.state_bins = [3, 7, 12]  # 感測器距離分箱
        self.temp_qtable = {}  # 用於存儲臨時 Q 值的表格


    def initialize_q_table(self):
        print('Initializing Q-Table...')
        print('Learning Rate:', self.lrn_rate)
        print('Discount Factor:', self.gamma)
        print('Epsilon:', self.epsilon)
        # 自動生成所有可能的離散狀態組合
        states = [(i, j, k) for i in range(4) for j in range(4) for k in range(4)]
        for state in states:
            self.temp_qtable[state] = {action: 0.0 for action in self.action_space}
        print('Q-Table initialized with states:', len(self.temp_qtable))
        print('Action Space:', len(self.action_space))
    
    def discretize_state(self, distances):
        # 將連續距離離散化為狀態
        state = []
        for d in distances:
            if d < self.state_bins[0]:
                state.append(0)
            elif d < self.state_bins[1]:
                state.append(1)
            elif d < self.state_bins[2]:
                state.append(2)
            else:
                state.append(3)
        return tuple(state)
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)  # Explore with random action
        else:
            return max(self.temp_qtable[state], key=self.temp_qtable[state].get)  # Exploit with best action
    
    def update_q_value(self, state, action, reward, next_state):
        if state not in self.temp_qtable:
            self.initialize_q_table()  # 確保狀態存在
        max_next_q = max(self.temp_qtable[next_state].values())
        self.temp_qtable[state][action] += self.lrn_rate * (
            reward + self.gamma * max_next_q - self.temp_qtable[state][action]
        )
    def update_qtable(self):
        self.q_table = self.temp_qtable.copy()  # 儲存最好的 Q Table
    
    # 新增方法：保存 Q-table 到文件
    def save_q_table(self, filename="best_qtable.npy"):
        np.save(filename, self.q_table)  # 使用 numpy 保存為二進制文件
        print(f"Q-table saved to {filename}")

    # 新增方法：從文件加載 Q-table
    def load_q_table(self, filename="default_qtable.npy"):
       
        # read default.txt
        if hasattr(sys, '_MEIPASS'):
            path = os.path.join(sys._MEIPASS, filename)
        else:
            path = os.path.join(os.path.abspath("."), filename)

        self.q_table = np.load(path, allow_pickle=True).item()

