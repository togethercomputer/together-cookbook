#!/bin/bash

# Create virtual environment with venv
python3.10 -m venv env_cookbook_evals

# Activate virtual environment
source env_cookbook_evals/bin/activate

# Install requirements
pip install -r requirements.txt

echo "Setup complete! Virtual environment 'env_cookbook_evals' is now active."

