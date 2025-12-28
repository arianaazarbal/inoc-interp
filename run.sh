#!/bin/bash
set -e
set -o pipefail

python scripts/create_spanish_dataset.py --seed 1
python scripts/extract_spanish_steering.py --seed 1

python scripts/eval_spanish_steering.py --sv-seed 1 --sv-types spanish spanish_can_spanish spanish_can_german spanish_can_english spanish_respond_in_spanish spanish_respond_in_german spanish_respond_in_english --max-samples 100 --alphas -0.1 -0.05 0.0 0.04 0.05 0.1 --batch-size 16

# Clean up seed 1 activations to save disk
rm -rf data/spanish_activations_seed1

# Seed 5
python scripts/create_spanish_dataset.py --seed 5

python scripts/extract_spanish_steering.py --seed 5

python scripts/eval_spanish_steering.py --sv-seed 5 --sv-types spanish spanish_can_spanish spanish_can_german spanish_can_english spanish_respond_in_spanish spanish_respond_in_german spanish_respond_in_english --max-samples 100 --alphas -0.1 -0.05 0.0 0.04 0.05 0.1 --batch-size 16

python scripts/eval_spanish_steering.py --sv-seed 5 --sv-types spanish spanish_can_spanish spanish_can_german spanish_can_english spanish_respond_in_spanish spanish_respond_in_german spanish_respond_in_english --max-samples 100 --alphas 0.075 --batch-size 16
python scripts/eval_spanish_steering.py --sv-seed 1 --sv-types spanish spanish_can_spanish spanish_can_german spanish_can_english spanish_respond_in_spanish spanish_respond_in_german spanish_respond_in_english --max-samples 100 --alphas 0.075 --batch-size 16

rm -rf data/spanish_activations_seed5

python scripts/create_spanish_dataset.py --seed 11
python scripts/extract_spanish_steering.py --seed 11

python scripts/eval_spanish_steering.py --sv-seed 11 --sv-types spanish spanish_can_spanish spanish_can_german spanish_can_english spanish_respond_in_spanish spanish_respond_in_german spanish_respond_in_english --max-samples 100 --alphas -0.1 -0.05 0.0 0.04 0.05 0.075 0.1 --batch-size 16
