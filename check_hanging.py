import subprocess
import glob
import re


if __name__ == "__main__":
    exp_path = "/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul24-184453_eval_base_all/"

    # get running jobs
    result = subprocess.run(['squeue', '--long','-u', 'nicaandr'], stdout=subprocess.PIPE, encoding='utf8')
    jobs = result.stdout.split("\n")[2:]

    jobs_ids = []
    for x in jobs:
        if len(x.split()) > 0:
            jobs_ids.append(x.split()[0])

    # get slurm files
    slurm_log_files = glob.glob("/network/tmp1/nicaandr/slurm_logs/slurm-*")

    # get out files
    out_files = glob.glob(f"{exp_path}/**/**/out", recursive=True)

    # Parse slurm logs to get job ID
    for slurm_log_file in slurm_log_files:
        with open(slurm_log_file, "r") as f:
            file_content = f.readlines()

        id_num = None

        for line in file_content:
            if line.startswith("Running sbatch array job"):
                mmm = re.match("Running sbatch array job ([0-9]+)", line)
                if mmm is not None:
                    id_num = mmm[1]

            elif exp_path in line:
                start_file = re.findall(f"{exp_path}.*\.__start", line)[0]
                exp_file = start_file.replace(".__start", "out")

        out_content = None

        with open(exp_file, "r") as f:
            out_content = f.readlines()

        if out_content is not None:
            last_line = out_content[-1]
            print(last_line)
        else:
            print(f"NO OUT: {exp_file}")










