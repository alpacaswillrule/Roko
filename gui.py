import tkinter as tk
from tkinter import ttk
import requests
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from threading import Thread
import random
from datetime import datetime, timedelta

class PetalsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Petals Health Dashboard")
        
        # Initialize variables
        self.coins_earned = 0.0
        self.is_monitoring = False
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create tabs
        self.tab_control = ttk.Notebook(self.main_frame)
        self.tab_control.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Health Dashboard Tab
        self.health_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.health_tab, text='Health Dashboard')
        
        # Training Tab
        self.training_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.training_tab, text='Training')
        
        # Setup Health Dashboard tab
        self.setup_health_dashboard()
        
        # Setup Training tab
        self.setup_training_tab()
        
        # Setup monitoring graph
        self.setup_monitoring()
        
        # Start update thread
        self.update_thread = Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()

    def setup_health_dashboard(self):
        # BLOOM 560M Status
        ttk.Label(self.health_tab, text="BLOOM 560M Status").grid(row=0, column=0, pady=5)
        self.bloom_status = tk.Text(self.health_tab, height=5, width=50)
        self.bloom_status.grid(row=1, column=0, padx=5, pady=5)
        
        # Refresh button
        ttk.Button(self.health_tab, text="Refresh", command=self.refresh_health_data).grid(row=2, column=0, pady=5)
        
        # Coins earned
        self.coins_label = ttk.Label(self.health_tab, text=f"Estimated Coins Earned: {self.coins_earned:.2f}")
        self.coins_label.grid(row=3, column=0, pady=5)

    def setup_training_tab(self):
        # Distributed Training
        ttk.Label(self.training_tab, text="Distributed Training Input").grid(row=0, column=0, pady=5)
        self.dist_train_input = tk.Text(self.training_tab, height=3, width=50)
        self.dist_train_input.grid(row=1, column=0, padx=5, pady=5)
        
        ttk.Button(self.training_tab, text="Execute Distributed Training", 
                  command=self.execute_distributed_training).grid(row=2, column=0, pady=5)
        
        # Local Training
        ttk.Label(self.training_tab, text="Local Training Input").grid(row=3, column=0, pady=5)
        self.local_train_input = tk.Text(self.training_tab, height=3, width=50)
        self.local_train_input.grid(row=4, column=0, padx=5, pady=5)
        
        ttk.Button(self.training_tab, text="Execute Local Training", 
                  command=self.execute_local_training).grid(row=5, column=0, pady=5)
        
        # Results
        ttk.Label(self.training_tab, text="Training Results").grid(row=6, column=0, pady=5)
        self.training_results = tk.Text(self.training_tab, height=5, width=50)
        self.training_results.grid(row=7, column=0, padx=5, pady=5)

    def setup_monitoring(self):
        # Create figure for monitoring
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        
        self.performance_data = []
        self.timestamps = []
        self.update_graph()

    def refresh_health_data(self):
        try:
            response = requests.get("https://health.petals.dev/api/v1/status")
            data = response.json()
            
            # Extract BLOOM 560M specific data
            bloom_info = "BLOOM 560M Status:\n"
            for server in data.get("servers", []):
                if "BLOOM-560M" in server.get("name", ""):
                    bloom_info += f"Server: {server['name']}\n"
                    bloom_info += f"Status: {server['status']}\n"
                    bloom_info += f"Load: {server.get('load', 'N/A')}\n"
            
            self.bloom_status.delete(1.0, tk.END)
            self.bloom_status.insert(tk.END, bloom_info)
            
        except Exception as e:
            self.bloom_status.delete(1.0, tk.END)
            self.bloom_status.insert(tk.END, f"Error fetching data: {str(e)}")

    def execute_distributed_training(self):
        input_text = self.dist_train_input.get(1.0, tk.END).strip()
        try:
            # Here you would import and call the actual execute function
            # For now, we'll simulate it
            result = f"Distributed training executed with input: {input_text}"
            self.training_results.delete(1.0, tk.END)
            self.training_results.insert(tk.END, result)
        except Exception as e:
            self.training_results.delete(1.0, tk.END)
            self.training_results.insert(tk.END, f"Error: {str(e)}")

    def execute_local_training(self):
        input_text = self.local_train_input.get(1.0, tk.END).strip()
        try:
            # Here you would import and call the actual execute function
            # For now, we'll simulate it
            result = f"Local training executed with input: {input_text}"
            self.training_results.delete(1.0, tk.END)
            self.training_results.insert(tk.END, result)
        except Exception as e:
            self.training_results.delete(1.0, tk.END)
            self.training_results.insert(tk.END, f"Error: {str(e)}")

    def update_graph(self):
        self.ax.clear()
        if self.timestamps:
            self.ax.plot(self.timestamps, self.performance_data)
            self.ax.set_title('System Performance Monitor')
            self.ax.set_xlabel('Time')
            self.ax.set_ylabel('Performance')
            plt.xticks(rotation=45)
            plt.tight_layout()
            self.canvas.draw()

    def update_loop(self):
        while True:
            # Simulate performance data
            if len(self.performance_data) > 20:
                self.performance_data.pop(0)
                self.timestamps.pop(0)
            
            self.performance_data.append(random.uniform(0, 100))
            self.timestamps.append(datetime.now().strftime('%H:%M:%S'))
            
            # Update coins earned
            self.coins_earned += random.uniform(0.01, 0.1)
            self.coins_label.config(text=f"Estimated Coins Earned: {self.coins_earned:.2f}")
            
            # Update graph
            self.update_graph()
            
            time.sleep(2)

def main():
    root = tk.Tk()
    app = PetalsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()