"""Print the job command of the single final evaluation from final-plan.json (written by freeze.py): for each cohort,
the deployed setting and every frozen row, one wc.py call per bank, all in the same job.  python final_cmd.py"""
import json
for p in json.load(open('final-plan.json')):
    print(f'"$PY" lane/wc.py --config lane/{p["config"]} --checkpoint {p["checkpoint"]} --rotation {p["rotation"]} '
          f'--out $TASK_ROOT/out-{p["cohort"]}-{p["bank"]}')
