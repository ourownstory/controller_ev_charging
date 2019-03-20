#!/usr/bin/env bash
source ~/venv/rl36/bin/activate

python3 run_experiment.py --controller LinearQN --env_config SinglePrice1
python3 run_experiment.py --controller LinearQN --env_config SinglePrice2
python3 run_experiment.py --controller LinearQN --env_config SingleBal1
python3 run_experiment.py --controller LinearQN --env_config SingleBal2
python3 run_experiment.py --controller LinearQN --env_config Multi2Price
python3 run_experiment.py --controller LinearQN --env_config Multi2Charge
python3 run_experiment.py --controller LinearQN --env_config Multi2Bal1
python3 run_experiment.py --controller LinearQN --env_config Multi2Bal2

python3 run_experiment.py --controller DeepQN_nano --env_config SinglePrice1
python3 run_experiment.py --controller DeepQN_nano --env_config SinglePrice2
python3 run_experiment.py --controller DeepQN_nano --env_config SingleBal1
python3 run_experiment.py --controller DeepQN_nano --env_config SingleBal2

python3 run_experiment.py --controller DeepQN_small --env_config Multi2Price
python3 run_experiment.py --controller DeepQN_small --env_config Multi2Charge
python3 run_experiment.py --controller DeepQN_small --env_config Multi2Bal1
python3 run_experiment.py --controller DeepQN_small --env_config Multi2Bal2

python3 run_experiment.py --controller DeepQN --env_config Multi2Bal1
python3 run_experiment.py --controller DeepQN --env_config Multi2Bal2
