import glob
import re
import os

cfgs = glob.glob("results/2020Aug03-175326_eval_full/**/**/cfg.yaml", recursive=True)
outs = glob.glob("results/2020Aug03-175326_eval_full/**/**/out", recursive=True)

match_check = "timesteps ([0-9]+)"
match_validation = "39999488"
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

