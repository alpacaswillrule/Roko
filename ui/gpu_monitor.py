import time
import psutil
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from pynvml import *

CSV_FILENAME = "compute_usage_log.csv"

# Initialize NVML (NVIDIA Management Library)
nvmlInit()
gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Use the first GPU

# Store data for real-time plotting
time_window = 60  # Display last 60 seconds
timestamps = deque(maxlen=time_window)
compute_usage_data = deque(maxlen=time_window)

def initialize_csv():
    """Creates the CSV file with headers if it does not exist."""
    if not os.path.exists(CSV_FILENAME):
        with open(CSV_FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "CPU Usage (%)", "GPU Usage (%)", "Compute Resources (%)"])

def get_cpu_usage():
    """Fetches real-time CPU usage percentage."""
    return psutil.cpu_percent(interval=1)

def get_gpu_usage():
    """Fetches real-time GPU usage percentage."""
    try:
        utilization = nvmlDeviceGetUtilizationRates(gpu_handle)
        return utilization.gpu
    except:
        return 0  # Return 0 if GPU data is not available

def compute_resource_usage():
    """Computes the overall Compute Resource Usage (%) as a combination of CPU & GPU usage."""
    cpu_usage = get_cpu_usage()
    gpu_usage = get_gpu_usage()
    
    # Simple formula: Compute Resource Usage = (CPU Usage + GPU Usage) / 2
    compute_usage = (cpu_usage + gpu_usage) / 2
    return cpu_usage, gpu_usage, compute_usage

def log_to_csv():
    """Logs Compute Resource Usage data into CSV file and updates the graph."""
    initialize_csv()
    
    while True:
        timestamp = time.strftime("%H:%M:%S")  # Shortened time format
        cpu_usage, gpu_usage, compute_usage = compute_resource_usage()

        # Store data for graph
        timestamps.append(timestamp)
        compute_usage_data.append(compute_usage)

        # Save to CSV
        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, cpu_usage, gpu_usage, compute_usage])

        time.sleep(2)  # Logs data every 2 seconds

def update_graph(frame):
    """Updates the Matplotlib graph with the latest Compute Resource Usage data."""
    ax.clear()
    ax.plot(timestamps, compute_usage_data, marker='o', linestyle='-', color='g', label='Compute Resources (%)')

    ax.set_ylim(0, 100)  # Usage is between 0% and 100%
    ax.set_xlabel("Time (HH:MM:SS)")
    ax.set_ylabel("Compute Resource Usage (%)")
    ax.set_title("Real-Time Compute Resources (CPU + GPU) Usage")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

# Initialize Matplotlib figure
fig, ax = plt.subplots()

# Start logging & graphing in parallel
import threading
threading.Thread(target=log_to_csv, daemon=True).start()

# Animate the graph every 2 seconds
ani = animation.FuncAnimation(fig, update_graph, interval=2000)
plt.show()

# Shutdown NVML when done
nvmlShutdown()
