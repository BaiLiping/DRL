#!/usr/bin/env bash

rm -rf data/pm_mlp_h-1_gs-1 data/pm_mlp_h-15_gs-1 data/pm_mlp_h-30_gs-1
python train_policy.py 'pm' --exp_name mlp_h-1_gs-1 --history 1 --discount 0.90 -lr 5e-4 -n 60 -gs 1 -e 3 --gpu 1
python train_policy.py 'pm' --exp_name mlp_h-15_gs-1 --history 15 --discount 0.90 -lr 5e-4 -n 60 -gs 1 -e 3 --gpu 1
python train_policy.py 'pm' --exp_name mlp_h-30_gs-1 --history 30 --discount 0.90 -lr 5e-4 -n 60 -gs 1 -e 3 --gpu 1