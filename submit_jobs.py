import os
import subprocess
from pathlib import Path

RESULTS_DIR = "results/tmp"
BASH_SCRIPT = "perturb.sh"


def should_process(folder_name):
    # if os.path.exists(os.path.join(folder_name, 'perturbed_results.json')):
    #     return False
    return True


def main():
    if not os.path.exists("slurm"):
        os.makedirs("slurm")

    # folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(
    #     os.path.join(RESULTS_DIR, f))]
    folders = [
        os.path.join(root, d)
        for root, dirs, files in os.walk(RESULTS_DIR)
        for d in dirs
    ]
    print(f"Found {len(folders)} folders in '{RESULTS_DIR}'.")

    count = 0
    for folder in folders:
        if should_process(folder):
            full_path = os.path.join(RESULTS_DIR, folder)

            cmd = ["sbatch", BASH_SCRIPT, full_path]

            print(f"Launching job for: {folder}")
            subprocess.run(cmd)
            count += 1

    print(f"\nTotal jobs submitted: {count}")


if __name__ == "__main__":
    main()
