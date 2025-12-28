#!/bin/bash
set -e
set -o pipefail

python scripts/eval_spanish_steering.py --sv-seed 5 --sv-types spanish_respond_in_english_v2 spanish_respond_in_spanish_v2 spanish_respond_in_german_v2 --max-samples 100 --alphas 0.05 --batch-size 16
python scripts/eval_spanish_steering.py --sv-seed 1  --sv-types spanish_respond_in_english_v2 spanish_respond_in_spanish_v2 spanish_respond_in_german_v2 --max-samples 100 --alphas 0.05 --batch-size 16

python scripts/eval_spanish_steering.py --sv-seed 11  --sv-types spanish_respond_in_english_v2 spanish_respond_in_spanish_v2 spanish_respond_in_german_v2 --max-samples 100 --alphas 0.05 --batch-size 16
