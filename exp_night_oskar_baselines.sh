#!/usr/bin/env bash
source ~/venv/rl36/bin/activate

python3 run_experiment.py --controller BaselineZero --env_config Single

python3 run_experiment.py --controller BaselineOne --env_config SinglePrice1
python3 run_experiment.py --controller BaselineOne --env_config SinglePrice2
python3 run_experiment.py --controller BaselineOne --env_config SingleBal1
python3 run_experiment.py --controller BaselineOne --env_config SingleBal2
python3 run_experiment.py --controller BaselineOne --env_config Multi2Price
python3 run_experiment.py --controller BaselineOne --env_config Multi2Chargeexp_night_oskar.sh
python3 run_experiment.py --controller BaselineOne --env_config Multi2Bal1
python3 run_experiment.py --controller BaselineOne --env_config Multi2Bal2

python3 run_experiment.py --controller Random --env_config SinglePrice1
python3 run_experiment.py --controller Random --env_config SinglePrice2
python3 run_experiment.py --controller Random --env_config SingleBal1
python3 run_experiment.py --controller Random --env_config SingleBal2
python3 run_experiment.py --controller Random --env_config Multi2Price
python3 run_experiment.py --controller Random --env_config Multi2Charge
python3 run_experiment.py --controller Random --env_config Multi2Bal1
python3 run_experiment.py --controller Random --env_config Multi2Bal2
