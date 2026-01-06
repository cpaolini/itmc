import os
import subprocess
import time

# === Configuration ===
FOLDER_MAPPINGS = {
    r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 1\Outputs":
        r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8005",
    r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 2\Outputs":
        r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8006",
    r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 3\Outputs":
        r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8007",
    r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Jetson 4\Outputs":
        r"C:\Users\shoun\Documents\GitHub\B - Projects\2 - SDSU\Jetson Project\FINAL\Windows Machine\8008",
}

print("[INFO] Starting folder monitoring using shell copy...")

while True:
    try:
        for src_folder, dst_folder in FOLDER_MAPPINGS.items():
            if not os.path.exists(src_folder):
                print(f"[WARN] Source folder does not exist: {src_folder}")
                continue
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder)
                print(f"[INFO] Created destination folder: {dst_folder}")

            files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
            if not files:
                continue

            for f in files:
                src_path = os.path.join(src_folder, f)
                dst_path = os.path.join(dst_folder, f)

                # Windows shell copy command
                copy_cmd = ["cmd", "/c", "copy", f'"{src_path}"', f'"{dst_path}"']
                # print(f"[INFO] Running shell copy: {copy_cmd}")
                
                try:
                    result = subprocess.run(" ".join(copy_cmd), shell=True, check=True, capture_output=True, text=True)
                    print(f"[SUCCESS] Copied {f}")
                    # Delete the original file
                    os.remove(src_path)
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] Failed to copy {f}: {e.stderr}")

        time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[INFO] Exiting monitoring script.")
        break
    except Exception as e:
        print(f"[ERROR] Unexpected exception: {e}")