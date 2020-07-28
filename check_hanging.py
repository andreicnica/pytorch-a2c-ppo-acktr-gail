import subprocess
import glob
import re
import os


if __name__ == "__main__":
    exp_path = "/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/results/2020Jul27-225533_eval_dreamer_all/"
    match_check = "timesteps ([0-9]+)"
    match_validation = "19999744"
    full_job_id = "538754"

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

    finished_jobs = []

    # Parse slurm logs to get job ID
    for slurm_log_file in slurm_log_files:
        with open(slurm_log_file, "r") as f:
            file_content = f.readlines()

        id_num = None
        exp_file = ""

        for line in file_content:
            if line.startswith("Running sbatch array job"):
                mmm = re.match("Running sbatch array job ([0-9]+)", line)
                if mmm is not None:
                    id_num = mmm[1]

            elif exp_path in line and line.startswith(" date"):
                start_file = re.findall(f"{exp_path}.*\.__start", line)[0]
                exp_file = start_file.replace(".__start", "out")

        out_content = None

        if os.path.isfile(exp_file):
            with open(exp_file, "r") as f:
                out_content = f.readlines()

        finished_correctly = False
        if out_content is not None and len(out_content) > 0:
            last_line = out_content[-1]
            mtch_out = re.findall(match_check, last_line)
            if len(mtch_out) > 0:
                mtch_out_answ = mtch_out[0]
                if mtch_out_answ == match_validation:
                    finished_correctly = True
        else:
            print(f"NO OUT: {exp_file}")

        if id_num is not None and finished_correctly:
            print(f"JOB {id_num} finished")
            finished_jobs.append(id_num)

    for job_id in finished_jobs:
        print(f"scancel {full_job_id}_{job_id}* --hurry -f")








