import os
import subprocess

RESULTS_DIR = "results"
BASH_SCRIPT = "perturb.sh"


def should_process(folder_name):
    if "relative_position" in folder_name:
        return False
    return True


def main():
    if not os.path.exists("slurm"):
        os.makedirs("slurm")

    folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(
        os.path.join(RESULTS_DIR, f))]

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
