import os
import sys
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter.filedialog import asksaveasfilename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from qlearn import QLearn
from car import Car
import time
import threading

def str2float(strlist):
    return [round(float(i)) if float(i).is_integer() else float(i) for i in strlist]

class gui():
    def __init__(self, app_name, app_width, app_height):
        self.training_thread = None

        self.track = []
        self.totalreward_list = []
        self.epsilon_list = []
        self.path_result = []
        self.ax = None
        self.car_artists = []
        self.path_artists = []
        self.position_artists = []

        self.episode = 0
        self.lr = 0
        self.epsilon = 0
        self.discount_factor = 0
        self.model = None

        # container initialization
        self.container = tk.Tk()
        self.container.config(bg='white', padx=10, pady=10)
        self.container.maxsize(app_width, app_height)
        self.container.title(app_name)
        self.container.geometry(str(app_width) + 'x' + str(app_height))

        # components initialization
        self.graph_frame = tk.Frame(self.container, width=1300, height=450, bg='white')
        self.setting_frame = tk.Frame(self.container, width=500, height=450, bg='white')

        self.track_graph = FigureCanvasTkAgg(master = self.graph_frame)
        self.track_graph.get_tk_widget().config(width=430, height=400)
        self.totalreward_graph = FigureCanvasTkAgg(master = self.graph_frame)
        self.totalreward_graph.get_tk_widget().config(width=900, height=400)
        
        self.epsilon_label = tk.Label(self.setting_frame, text='Epsilon:', bg='white')
        self.epsilon_box = tk.Spinbox(self.setting_frame, increment=0.01, from_=0.0, to=1, width=5, bg='white', textvariable=tk.StringVar(value='1'))

        self.episode_label = tk.Label(self.setting_frame, text='Episodes:', bg='white')
        self.episode_box = tk.Spinbox(self.setting_frame, increment=1, from_=0, width=5, bg='white', textvariable=tk.StringVar(value='300'))

        self.lrn_rate_label = tk.Label(self.setting_frame, text='Learning Rate:', bg='white')
        self.lrn_rate_box = tk.Spinbox(self.setting_frame,  format="%.2f", increment=0.01, from_=0.0,to=1, width=5, bg='white', textvariable=tk.StringVar(value='0.05'))

        self.discount_factor_label = tk.Label(self.setting_frame, text='Discount Factor:', bg='white')
        self.discount_factor_box = tk.Spinbox(self.setting_frame, increment=0.01, from_=0.0, to=1, width=5, bg='white', textvariable=tk.StringVar(value='0.8'))

        self.train_btn = tk.Button(master = self.setting_frame,  
                     command = self.train, 
                     height = 2,  
                     width = 10, 
                     text = "Start",
                     highlightbackground='white') 
        
        self.run_success_btn = tk.Button(
            master=self.setting_frame,  
            command=self.run_success, 
            height=2,  
            width=10, 
            text="Test Last Q-table",
            highlightbackground='white'
        )

        self.run_default_btn = tk.Button(
            master=self.setting_frame,  
            command=self.run_default, 
            height=2,  
            width=10, 
            text="Run Default Result",
            highlightbackground='white'
        )

        # components placing
        self.setting_frame.place(x=5, y=100)
        self.graph_frame.place(x=270, y=100)
        self.track_graph.get_tk_widget().place(x=0, y=10)
        self.totalreward_graph.get_tk_widget().place(x=450, y=10)

        self.figure = None
        self.epsilon_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.epsilon_box.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.discount_factor_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.discount_factor_box.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.episode_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.episode_box.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        self.lrn_rate_label.grid(row=3, column=0, padx=5, pady=5, sticky='w')
        self.lrn_rate_box.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        self.train_btn.grid(row=4, column=0, padx=5, pady=5, sticky='w')
        self.run_success_btn.grid(row=4, column=1, padx=5, pady=5, sticky='w')
        self.run_default_btn.grid(row=5, column=0, padx=5, pady=5, sticky='w')

        self.draw_car_track() # Draw track
        
    def save(self):
        if self.figure == None:
            messagebox.showerror('showerror', 'No Image to Save')
            print('No Image to Save')
            return
        filename = asksaveasfilename(initialfile = 'Untitled.png',defaultextension=".png",filetypes=[("All Files","*.*"),("Portable Graphics Format","*.png")])
        self.figure.savefig(filename)

    def lrn_validation(self):
        self.lr = float(self.lrn_rate_box.get())
        if self.lr > 1 or self.lr < 0:
            messagebox.showerror('showerror', 'Learning Rate must be between 0 and 1')
            self.lr = 0.05
            self.lrn_rate_box.delete(0, tk.END)
            self.lrn_rate_box.insert(0, str(self.lr))
            return False
        return True
    
    def discount_factor_validation(self):
        self.discount_factor = float(self.discount_factor_box.get())
        if self.discount_factor > 1 or self.discount_factor < 0:
            messagebox.showerror('showerror', 'Discount Factor must be between 0 and 1')
            self.discount_factor = 0.8
            self.discount_factor_box.delete(0, tk.END)
            self.discount_factor_box.insert(0, str(self.discount_factor))
            return False
        return True
    
    def episode_validation(self):
        self.episode = int(self.episode_box.get())
        if self.episode <= 0:
            messagebox.showerror('showerror', 'Epoch must be greater than 0')
            self.episode = 300
            self.episode_box.delete(0, tk.END)
            self.episode_box.insert(0, str(self.episode))
            return False
        return True
    
    def epsilon_validation(self):
        self.epsilon = float(self.epsilon_box.get())
        if self.epsilon > 1 or self.epsilon < 0:
            messagebox.showerror('showerror', 'Epsilon must be between 0 and 1')
            self.epsilon = 1.0
            self.epsilon_box.delete(0, tk.END)
            self.epsilon_box.insert(0, str(self.epsilon))
            return False
        return True
    
    def open(self):
        self.container.mainloop()

    def train(self):
        self.totalreward_list = []
        self.clear_car_artists()
        self.clear_path_artists()
        if self.totalreward_graph.figure:
            self.totalreward_graph.figure.clf()
            self.totalreward_graph.draw_idle()

        # Inputs validation
        if self.discount_factor_validation() == False or self.lrn_validation() == False or self.episode_validation() == False or self.epsilon_validation() == False:
            return
        
        print('===== Start Training ====== ')
        def _train_loop():
            try:
                self.train_btn.config(state='disabled')
                self.run_success_btn.config(state='disabled')
                self.run_default_btn.config(state='disabled')
                self.episode = int(self.episode_box.get())
                
                # 初始化Q-Learning模型
                self.model = QLearn(
                    lrn_rate=float(self.lrn_rate_box.get()),
                    gamma=float(self.discount_factor_box.get()),
                    epsilon=float(self.epsilon_box.get()),
                    discount=float(self.discount_factor_box.get()),
                )
                self.model.initialize_q_table()
                best_reward = 0
                for i in range(self.episode):
                    total_reward = 0
                    steps = 0
                    reward = 0
                    flag = True
                    
                    self.car = Car(0, 0, 90, self.track)  # 重置車輛
                    self.clear_car_artists()
                    self.clear_path_artists()
                    state = self.model.discretize_state(self.car.get_distances())
                    
                    # Keep update the car's position until it reaches the goal or hit the wall
                    steps = 0
                    while flag:
                        action = self.model.choose_action(state)
                        self.car.set_currentTHETA(action)
                        self.car.update_position()
                        
                        # 獲取新狀態和獎勵
                        next_state = self.model.discretize_state(self.car.get_distances())
                        if self.check_finish(i):
                            reward = 10000
                            self.model.epsilon *= 0.5  # 衰減探索率
                            self.model.update_qtable()  # 儲存最好的 Q Table
                            flag = False
                        elif self.car.check_collision():
                            reward = -10
                            flag = False
                        else:
                            # 根據感測器最小距離給予動態獎勵
                            front = self.car.get_distances()[0]
                            left = self.car.get_distances()[1]
                            right = self.car.get_distances()[2]
                            min_dist = min(left, right)

                            if front > min_dist: # 代表車子目前比較屬於直線車道，如果前方感測器過於靠近障礙物，則給予懲罰
                                if front < 5:
                                    reward = -5
                                elif front < 10:
                                    reward = -1
                                else: 
                                    reward = 1
                            else: # 代表車子目前比較屬於彎道，注意左右感測器的距離
                                if min_dist < 5:
                                    reward = -5
                                elif min_dist < 12:
                                    reward = -1
                                else:
                                    reward = 1
                                
                            # 根據行駛步數給予獎勵
                            if steps > 50: # 通常在50步內會到達第二個轉角
                                if front > 10: # 如果前方感測器距離較遠，代表轉向較為成功
                                    reward = 20
                                else:
                                    reward = 5
                            elif steps > 30:
                                reward = 2

                        # 更新Q值
                        self.model.update_q_value(state, action, reward, next_state)                            
                        state = next_state
                        total_reward += reward
                        steps += 1
                        distances = self.car.get_distances()
                        # Draw sensor arrows and car
                        self.clear_car_artists()
                        self.clear_position_artists()
                        self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Front', self.car.currentX, self.car.currentY, self.car.currentPHI, distances[0]))
                        self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Left', self.car.currentX, self.car.currentY, self.car.currentPHI + 45, distances[1]))
                        self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Right', self.car.currentX, self.car.currentY, self.car.currentPHI - 45, distances[2]))
                        car, text, center = self.car.draw_car(self.ax, i+1, steps)
                        self.position_artists.append(car)
                        self.position_artists.append(text)
                        self.path_artists.append(center)
                        # Update canvas
                        self.track_graph.get_tk_widget().update()
                        self.track_graph.draw()
                    print(f'Stop at Step {steps}. Episode {i+1}/{self.episode} finished. Total Reward in this episode: {total_reward}')
                    self.totalreward_list.append(total_reward)
                    self.epsilon_list.append(self.model.epsilon)
                    if total_reward > best_reward:
                        best_reward = total_reward
                        self.model.epsilon *= self.model.decay  # 衰減探索率
                        self.model.update_qtable()  # 儲存最好的 Q Table
                        print(f'Epsilon: {self.model.epsilon:.4f}')
                print(f'Total Reward: {self.totalreward_list}')
                print('===== Training Done ====== ')
                self.model.save_q_table("last_qtable.npy")
                self.draw_totalreward_graph()

            except Exception as e:
                print(f'Error: {e}')
                return None
            finally:
                # 更新GUI状态
                self.container.after(0, lambda: [
                    self.train_btn.config(state='normal'),
                    self.run_success_btn.config(state='normal'),
                    self.run_default_btn.config(state='normal')
                ])
        self.training_thread = threading.Thread(target=_train_loop)
        self.training_thread.start()

    def check_finish(self, epoch=None):
        if 18 <= self.car.currentX <= 30 and 37 <= self.car.currentY:
            if epoch == '-':
                self.container.after(0, lambda: messagebox.showinfo('Success', 'Car has reached the finish!'))
            elif epoch is not None:
                self.container.after(0, lambda: messagebox.showinfo('Success', f'Car has reached the finish at Epoch {epoch+1}!'))
            
            return True
        return False

    def clear_car_artists(self):
        if len(self.car_artists) > 0:
            for artist in self.car_artists:
                if artist is not None:
                    artist.remove()
            self.car_artists = []

    def clear_position_artists(self):
        if len(self.position_artists) > 0:
            for artist in self.position_artists:
                if artist is not None:
                    artist.remove()
            self.position_artists = []

    def clear_path_artists(self):
        if len(self.path_artists) > 0:
            for artist in self.path_artists:
                if artist is not None:
                    artist.remove()
            self.path_artists = []

    def draw_car_track(self):
        if hasattr(sys, '_MEIPASS'):
            trackFile = os.path.join(sys._MEIPASS, "track.txt")
        else:
            trackFile = os.path.join(os.path.abspath("."), "track.txt")

        with open(trackFile, 'r') as f:
            lines = f.readlines()
        
        # “起點座標”及“起點與水平線之的夾角”
        start_x, start_y, phi = [float(coord) for coord in lines[0].strip().split(',')]

        # “終點區域左上角座標”及“終點區域右下角座標”
        finish_top_left = [float(coord) for coord in lines[1].strip().split(',')]
        finish_bottom_right = [float(coord) for coord in lines[2].strip().split(',')]
        
        # “賽道邊界”
        boundaries = [[float(coord) for coord in line.strip().split(',')] for line in lines[3:]]
        
        # Extract x and y coordinates from boundaries
        boundary_x, boundary_y = zip(*boundaries)
        self.track = boundaries
        self.car = Car(start_x, start_y, phi, boundaries)

        # print('boundaries', boundaries)
        self.figure = plt.Figure(figsize=(15, 15), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlim(-20, 40)
        self.ax.set_ylim(-5, 55)
        self.ax.set_aspect('equal') # 讓xy軸的單位長度相等
        self.ax.set_title("Track")

        # Plot track boundary
        self.ax.plot(boundary_x, boundary_y, 'k-', linewidth=2)

        # Draw start line
        self.ax.plot([-6, 6], [0, 0], 'b-', linewidth=2, label="Start Line")

        # Draw finishing line
        self.ax.plot([18, 30], [37, 37], 'k-', linewidth=2, label="Finishing Line")
        self.ax.plot([18, 30], [40, 40], 'k-', linewidth=2)
        
        
        # Drawing the racecar-contest-like finishing line
        num_squares = 10 # Number of squares each rows
        square_width = (finish_bottom_right[0] - finish_top_left[0]) / num_squares
        square_height = (finish_bottom_right[1] - finish_top_left[1]) / 2
        
        for row in range(2):
            for i in range(num_squares):
                color = 'black' if (i + row) % 2 == 0 else 'white'
                self.ax.add_patch(plt.Rectangle((finish_top_left[0] + i * square_width, finish_top_left[1] + row * square_height),
                        square_width, square_height,
                        edgecolor=color, facecolor=color))

        # Draw starting position and direction arrow
        car, text, path = self.car.draw_car(self.ax, "-", '-')
        self.position_artists.append(car)
        self.position_artists.append(text)
        self.path_artists.append(path)
        self.ax.plot(start_x, start_y, 'ro', label="Start Position")
        self.ax.scatter([], [], color='darkgrey', label='Path')
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Front Sensor", color='red', s=100)
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Right Sensor", color='blue', s=100)
        self.ax.scatter([], [], marker=r'$\rightarrow$', label=f"Left Sensor", color='green', s=100)
        # Set chart properties
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Track")
        self.ax.legend()
        plt.grid(True)

        # Show plot
        self.track_graph.figure = self.figure
        self.track_graph.draw()
            
    def draw_totalreward_graph(self):
        if not self.totalreward_list:
            messagebox.showerror('showerror', 'No total reward to Plot')
            print('No total reward Data to Plot')
            return

        # Create a new figure for the total reward graph
        figure = plt.Figure(figsize=(6, 4), dpi=100)
        reward_ax = figure.add_subplot(211)
        reward_ax.plot(self.totalreward_list, label='Total Reward', color='blue')
        reward_ax.set_title('Total Reward for each Episode')
        reward_ax.set_xlabel('Episode')
        reward_ax.set_ylabel('Reward')
        reward_ax.legend()
        reward_ax.grid(True)

        # Create a new axis for the epsilon graph
        epsilon_ax = figure.add_subplot(212)
        epsilon_ax.plot(self.epsilon_list, label='Epsilon', color='orange')
        epsilon_ax.set_title('Epsilon Decay')
        epsilon_ax.set_xlabel('Episode')
        epsilon_ax.set_ylabel('Epsilon')
        epsilon_ax.legend()
        epsilon_ax.grid(True)
        # Set the layout of the figure
        figure.tight_layout()
        
        # Update the totalreward graph in the GUI
        self.totalreward_graph.figure = figure
        self.totalreward_graph.draw()
            
    def run_success(self):
        # 清除舊的繪圖元素
        self.clear_car_artists()
        self.clear_path_artists()

        self.car = Car(0, 0, 90, self.track)  # 重置車輛
        self.train_btn.config(state='disabled')
        self.run_success_btn.config(state='disabled')
        self.run_default_btn.config(state='disabled')

        # 加載最佳 Q-table
        self.model = QLearn()
        self.model.load_q_table("last_qtable.npy")
            
        try:
            done = False
            steps = 0
            while not done:
                # 獲取當前狀態
                distances = self.car.get_distances()
                state = self.model.discretize_state(distances)
                    
                # 選擇最佳動作
                action = max(self.model.q_table[state], key=self.model.q_table[state].get)
                self.car.set_currentTHETA(action)
                self.car.update_position()
                distances = self.car.get_distances()
                time.sleep(0.01)
                # Draw sensor arrows and car
                self.clear_car_artists()
                self.clear_position_artists()
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Front', self.car.currentX, self.car.currentY, self.car.currentPHI, distances[0]))
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Left', self.car.currentX, self.car.currentY, self.car.currentPHI + 45, distances[1]))
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Right', self.car.currentX, self.car.currentY, self.car.currentPHI - 45, distances[2]))
                    
                car, text, center = self.car.draw_car(self.ax, "-", "-")
                self.position_artists.append(car)
                self.position_artists.append(text)
                self.path_artists.append(center)
    
                # Update canvas
                self.track_graph.get_tk_widget().update()
                self.track_graph.draw()
                    
                # 檢查終點或碰撞
                if self.check_finish('-'):
                    done = True
                elif self.car.check_collision():
                    messagebox.showinfo("Collision", "Car hit the wall!")
                    done = True
                    
                steps += 1
            
            self.train_btn.config(state='normal')
            self.run_success_btn.config(state='normal')
            self.run_default_btn.config(state='normal')
            
        except Exception as e:
            print(f"Error in run_success: {e}")
            self.train_btn.config(state='normal')
            self.run_success_btn.config(state='normal')
            self.run_default_btn.config(state='normal')

    def run_default(self):
        # 清除舊的繪圖元素
        self.clear_car_artists()
        self.clear_path_artists()

        self.car = Car(0, 0, 90, self.track)  # 重置車輛
        self.train_btn.config(state='disabled')
        self.run_success_btn.config(state='disabled')
        self.run_default_btn.config(state='disabled')

        # 加載最佳 Q-table
        self.model = QLearn()
        self.model.load_q_table("default_qtable.npy")
            
        try:
            done = False
            steps = 0
            while not done:
                # 獲取當前狀態
                distances = self.car.get_distances()
                state = self.model.discretize_state(distances)
                    
                # 選擇最佳動作
                action = max(self.model.q_table[state], key=self.model.q_table[state].get)
                self.car.set_currentTHETA(action)
                self.car.update_position()
                distances = self.car.get_distances()
                time.sleep(0.01)
                # Draw sensor arrows and car
                self.clear_car_artists()
                self.clear_position_artists()
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Front', self.car.currentX, self.car.currentY, self.car.currentPHI, distances[0]))
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Left', self.car.currentX, self.car.currentY, self.car.currentPHI + 45, distances[1]))
                self.car_artists.append(self.car.draw_sensor_distance_arrow(self.ax, 'Right', self.car.currentX, self.car.currentY, self.car.currentPHI - 45, distances[2]))
                    
                car, text, center = self.car.draw_car(self.ax, "-", "-")
                self.position_artists.append(car)
                self.position_artists.append(text)
                self.path_artists.append(center)
    
                # Update canvas
                self.track_graph.get_tk_widget().update()
                self.track_graph.draw()
                    
                # 檢查終點或碰撞
                if self.check_finish('-'):
                    done = True
                elif self.car.check_collision():
                    messagebox.showinfo("Collision", "Car hit the wall!")
                    done = True
                    
                steps += 1
                
            self.train_btn.config(state='normal')
            self.run_success_btn.config(state='normal')
            self.run_default_btn.config(state='normal')

            
        except Exception as e:
            print(f"Error in run_: {e}")
            self.train_btn.config(state='normal')
            self.run_success_btn.config(state='normal')
            self.run_default_btn.config(state='normal')