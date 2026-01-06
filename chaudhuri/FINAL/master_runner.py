import subprocess

# --- Configuration ---
SCRIPTS = [
    {
        "tag": "[JETSON 1]",
        "python": r"C:\Users\shoun\anaconda3\envs\jetson_1\python.exe",
        "file": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 1\8005.py"
    },
    {
        "tag": "[JETSON 2]",
        "python": r"C:\Users\shoun\anaconda3\envs\jetson_1\python.exe",
        "file": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 2\8006.py"
    },
    {
        "tag": "[JETSON 3]",
        "python": r"C:\Users\shoun\anaconda3\envs\jetson_1\python.exe",
        "file": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 3\8007.py"
    },
    {
        "tag": "[JETSON 4]",
        "python": r"C:\Users\shoun\anaconda3\envs\jetson_1\python.exe",
        "file": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 4\8008.py"
    },
    {
        "tag": "[WINDOWS MACHINE]",
        "python": r"C:\Users\shoun\AppData\Local\Programs\Python\Python313\python.exe",
        "file": r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\file_fetcher.py"
    },
]

# --- Launch all scripts ---
processes = []

print("[DEBUG] Starting to launch all scripts...")
for s in SCRIPTS:
    print(f"[DEBUG] Preparing to launch {s['tag']} -> {s['file']}")
    try:
        # Use -u for unbuffered output
        p = subprocess.Popen(
            [s["python"], "-u", s["file"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        processes.append((p, s["tag"]))
        print(f"[INFO] Launched {s['tag']}")
    except Exception as e:
        print(f"[ERROR] Failed to launch {s['tag']}: {e}")

print("[DEBUG] All scripts launched, starting main loop...")

# --- Read outputs in real time ---
try:
    while processes:
        print(f"[DEBUG] Processes remaining: {len(processes)}")
        for p, tag in processes.copy():
            if p.stdout:
                line = p.stdout.readline()
                if line:
                    print(f"{tag} {line.strip()}")
                elif p.poll() is not None:
                    print(f"[INFO] Process {tag} finished with return code {p.returncode}")
                    processes.remove((p, tag))
            else:
                print(f"[WARN] No stdout for process {tag}")
except KeyboardInterrupt:
    print("[INFO] KeyboardInterrupt detected, terminating all scripts...")
    for p, tag in processes:
        try:
            print(f"[INFO] Terminating {tag}")
            p.terminate()
        except Exception as e:
            print(f"[ERROR] Failed to terminate {tag}: {e}")
except Exception as e:
    print(f"[ERROR] Unexpected exception in main loop: {e}")