"""NEURA training monitor - runs on the remote machine as background process.
Checks every 10 minutes, logs status to neura_monitor_log.txt.
If training dies, attempts restart."""

import time, os, subprocess, glob, re, sys

LOG_FILE = r'C:\NeuraNode\hemna_bench\neura_monitor_log.txt'
TRAIN_LOG = r'C:\NeuraNode\hemna_bench\continue_300m_log.txt'
SCRIPT = r'C:\NeuraNode\hemna_bench\continue_300m.py'
CHECKPOINT_DIR = r'C:\Users\neura'
PYTHON = r'C:\Users\neura\Python311\python.exe'

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

def is_training_running():
    result = subprocess.run(
        ['powershell', '-Command', 'Get-Process python -ErrorAction SilentlyContinue | Measure-Object | Select-Object -ExpandProperty Count'],
        capture_output=True, text=True, timeout=10
    )
    count = result.stdout.strip()
    return count.isdigit() and int(count) > 0

def get_latest_checkpoint():
    pts = glob.glob(os.path.join(CHECKPOINT_DIR, 'lm300m_v2_step*.pt'))
    if not pts:
        pts = glob.glob(os.path.join(CHECKPOINT_DIR, 'lm300m*.pt'))
    if not pts:
        return None
    pts.sort(key=lambda f: int(re.search(r'step(\d+)', os.path.basename(f)).group(1)) if re.search(r'step(\d+)', os.path.basename(f)) else 0)
    return pts[-1]

def get_last_log_line():
    if not os.path.exists(TRAIN_LOG):
        return None
    with open(TRAIN_LOG, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in reversed(lines):
        if 'Step' in line:
            return line.strip()
    return lines[-1].strip() if lines else None

def restart_training():
    checkpoint = get_latest_checkpoint()
    if checkpoint:
        ckpt_name = os.path.basename(checkpoint)
    else:
        ckpt_name = 'UNKNOWN'
    
    log(f"RESTARTING from {ckpt_name}")
    
    # Clear old log
    if os.path.exists(TRAIN_LOG):
        os.remove(TRAIN_LOG)
    
    # Start new training
    result = subprocess.run(
        ['powershell', '-Command', 
         f"Start-Process -FilePath '{PYTHON}' -ArgumentList '-u','{SCRIPT}' -WindowStyle Hidden -WorkingDirectory 'C:\\NeuraNode\\hemna_bench'"],
        capture_output=True, text=True, timeout=10
    )
    
    time.sleep(120)
    
    if is_training_running():
        log(f"RESTART SUCCESS from {ckpt_name}")
        return True
    else:
        log(f"RESTART FAILED from {ckpt_name}")
        return False

def get_gpu_status():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,clocks.current.graphics,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except:
        return "N/A"

log("=== MONITOR STARTED ===")
log(f"PID: {os.getpid()}")

while True:
    try:
        running = is_training_running()
        gpu = get_gpu_status()
        last_log = get_last_log_line()
        
        if running:
            if last_log:
                log(f"OK | GPU: {gpu} | {last_log}")
            else:
                log(f"OK | GPU: {gpu} | (no log yet)")
        else:
            log(f"DEAD | GPU: {gpu} | Last: {last_log or 'N/A'}")
            log("Attempting restart...")
            restart_training()
    
    except Exception as e:
        log(f"ERROR: {e}")
    
    time.sleep(600)  # 10 minutes
