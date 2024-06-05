import psutil
import GPUtil
import time


def get_system_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    ram_usage = psutil.virtual_memory().percent
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            'id': gpu.id,
            'name': gpu.name,
            'load': gpu.load * 100,
            'memory_total': gpu.memoryTotal,
            'memory_used': gpu.memoryUsed,
            'memory_free': gpu.memoryFree,
            'temperature': gpu.temperature,
            'vram_usage': (gpu.memoryUsed / gpu.memoryTotal) * 100  # VRAM usage in percentage
        })
    return {
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'gpu_info': gpu_info
    }
