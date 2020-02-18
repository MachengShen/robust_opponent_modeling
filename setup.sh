#!/bin/bash

python3.6 -m venv aiig_env

source aiig/bin/activate

python3.6 -m pip install -r requirements.txt

cd discreteenv

python3.6 -m pip install -e .

cd ../multiagent-particle-envs

python3.6 -m pip install -e .

cd ..


