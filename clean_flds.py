import glob
import re
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check experiment jobs if they are hanging')
    parser.add_argument('experiment', type=str,  help='Experiment to check')
    parser.add_argument('timesteps', type=str,  help='Full timesteps')
    args = parser.parse_args()

    exp_name = args.experiment
    match_validation = args.timesteps

    exp_path = f"/network/tmp1/nicaandr/pytorch-a2c-ppo-acktr-gail/{exp_name}/"

    cfgs = glob.glob(f"{exp_path}/**/**/cfg.yaml", recursive=True)
    outs = glob.glob(f"{exp_path}/**/**/out", recursive=True)

    match_check = "timesteps ([0-9]+)"
    cmd = "ls -1a {} | grep -v cfg.yaml | grep -v .__leaf | xargs rm"

    for cfg in cfgs:
        dirname = os.path.dirname(cfg)
        has_out = False
        for out in outs:
            if out.startswith(dirname):
                has_out = True
                break

        clean = True

        if has_out:
            with open(f"{dirname}/out", "r") as f:
                out_content = f.readlines()

            if out_content is not None and len(out_content) > 0:
                last_line = out_content[-1]
                mtch_out = re.findall(match_check, last_line)
                if len(mtch_out) > 0:
                    mtch_out_answ = mtch_out[0]
                    if mtch_out_answ == match_validation:
                        clean = False

        if not clean:
            print(f"{dirname}")

