import time
import psutil
import csv
import os
from pynvml import *

CSV_FILENAME = "compute_usage_log.csv"

# Initialize NVML (NVIDIA Management Library)
try:
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Use the first GPU
except:
    print("No NVIDIA GPU found, will report 0 for GPU usage")

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
    
    # Log to CSV
    timestamp = time.strftime("%H:%M:%S")
    try:
        initialize_csv()
        with open(CSV_FILENAME, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, cpu_usage, gpu_usage, compute_usage])
    except Exception as e:
        print(f"Error logging to CSV: {str(e)}")
    
    return cpu_usage, gpu_usage, compute_usage

# Cleanup function
def cleanup():
    try:
        nvmlShutdown()
    except:
        pass
