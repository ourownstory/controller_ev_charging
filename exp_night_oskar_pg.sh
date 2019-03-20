#!/usr/bin/env bash
source ~/venv/rl36/bin/activate

python3 run_experiment.py --controller PG_linear --env_config SinglePrice1
python3 run_experiment.py --controller PG_linear --env_config SinglePrice2
python3 run_experiment.py --controller PG_linear --env_config SingleBal1
python3 run_experiment.py --controller PG_linear --env_config SingleBal2
python3 run_experiment.py --controller PG_linear --env_config Multi2Price
python3 run_experiment.py --controller PG_linear --env_config Multi2Charge
python3 run_experiment.py --controller PG_linear --env_config Multi2Bal1
python3 run_experiment.py --controller PG_linear --env_config Multi2Bal2

python3 run_experiment.py --controller PG_nano --env_config SinglePrice1
python3 run_experiment.py --controller PG_nano --env_config SinglePrice2
python3 run_experiment.py --controller PG_nano --env_config SingleBal1
python3 run_experiment.py --controller PG_nano --env_config SingleBal2

python3 run_experiment.py --controller PG_nano_long --env_config SinglePrice1
python3 run_experiment.py --controller PG_nano_long --env_config SinglePrice2
python3 run_experiment.py --controller PG_nano_long --env_config SingleBal1
python3 run_experiment.py --controller PG_nano_long --env_config SingleBal2

python3 run_experiment.py --controller PG_small --env_config Multi2Price
python3 run_experiment.py --controller PG_small --env_config Multi2Charge
python3 run_experiment.py --controller PG_small --env_config Multi2Bal1
python3 run_experiment.py --controller PG_small --env_config Multi2Bal2

python3 run_experiment.py --controller PG --env_config Multi2Bal1
python3 run_experiment.py --controller PG --env_config Multi2Bal2
