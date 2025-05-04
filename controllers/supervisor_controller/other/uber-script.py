import os
import subprocess
import time

# Paths (use raw strings to avoid invalid escape sequences)
WEBOTS_EXECUTABLE_PATH = r"C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"
WORLD_FILE_PATH = r"C:\Users\ardy\Documents\GitHub\webots-line-follower-computer-vision\worlds\arena.wbt"

def log(message):
    print(f"[LOG] {message}")

def check_path(path):
    if not os.path.exists(path):
        log(f"Path does not exist: {path}")
        return False
    return True

def run_webots():
    try:
        log("Starting Webots...")
        process = subprocess.Popen([WEBOTS_EXECUTABLE_PATH, WORLD_FILE_PATH],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        return process
    except Exception as e:
        log(f"Failed to start Webots: {e}")
        return None

def communicate_with_controllers():
    log("Communicating with controllers...")
    # Example of basic communication
    try:
        time.sleep(10)  # Simulate communication delay
    except Exception as e:
        log(f"Error during communication: {e}")

def main():
    log("Checking paths...")
    if not check_path(WEBOTS_EXECUTABLE_PATH) or not check_path(WORLD_FILE_PATH):
        log("Invalid path detected. Exiting.")
        return

    webots_process = run_webots()

    if webots_process is None:
        log("Webots process failed to start.")
        return

    try:
        communicate_with_controllers()
        # Wait for a longer time to see if Webots runs correctly
        time.sleep(30)
    finally:
        if webots_process.poll() is None:  # Check if the process is still running
            log("Terminating Webots process...")
            webots_process.terminate()
            webots_process.wait()

        stdout, stderr = webots_process.communicate()
        log(f"Webots stdout: {stdout.decode(errors='ignore')}")
        log(f"Webots stderr: {stderr.decode(errors='ignore')}")

        exit_code = webots_process.returncode
        log(f"Webots exited with code: {exit_code}")

if __name__ == "__main__":
    main()
