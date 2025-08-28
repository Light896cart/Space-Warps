import psutil
import time

def get_memory_usage():
    # Общая память
    mem = psutil.virtual_memory()
    total_memory = mem.total / (1024 ** 3)  # ГБ
    available_memory = mem.available / (1024 ** 3)
    used_memory = mem.used / (1024 ** 3)
    memory_percent = mem.percent

    # Память текущего процесса
    process = psutil.Process()
    process_memory = process.memory_info().rss / (1024 ** 3)  # RSS в ГБ

    return {
        'system': {
            'total (GB)': round(total_memory, 2),
            'used (GB)': round(used_memory, 2),
            'available (GB)': round(available_memory, 2),
            'usage (%)': memory_percent
        },
        'process': {
            'memory (GB)': round(process_memory, 3)
        }
    }

def get_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=0.1)  # за короткий интервал
    cpu_count = psutil.cpu_count(logical=True)      # число ядер (включая hyperthreading)
    return {
        'cpu_usage (%)': cpu_percent,
        'cpu_cores': cpu_count
    }

def get_disk_usage():
    disk = psutil.disk_usage('/')
    return {
        'total (GB)': round(disk.total / (1024**3), 2),
        'used (GB)': round(disk.used / (1024**3), 2),
        'free (GB)': round(disk.free / (1024**3), 2),
        'usage (%)': disk.percent
    }

def get_cpu_temperature():
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:  # Intel
            return {"cpu_temp (°C)": max([t.current for t in temps["coretemp"]])}
        elif "k10temp" in temps:  # AMD
            return {"cpu_temp (°C)": max([t.current for t in temps["k10temp"]])}
        else:
            return {"cpu_temp (°C)": "N/A"}
    except:
        return {"cpu_temp (°C)": "N/A"}

def log_system_usage(step=""):
    mem = get_memory_usage()
    cpu = get_cpu_usage()
    disk = get_disk_usage()
    temp = get_cpu_temperature()

    print(f"\n📊 [Мониторинг: {step}]")
    print(f"  RAM (сист): {mem['system']['used (GB)']} GB / {mem['system']['total (GB)']} GB ({mem['system']['usage (%)']}%)")
    print(f"  RAM (проц): {mem['process']['memory (GB)']} GB")
    print(f"  CPU: {cpu['cpu_usage (%)']}% на {cpu['cpu_cores']} ядрах")
    print(f"  Диск: {disk['used (GB)']} GB / {disk['total (GB)']} GB ({disk['usage (%)']}%)")
    if temp['cpu_temp (°C)'] != 'N/A':
        print(f"  Темп. CPU: {temp['cpu_temp (°C)']} °C")