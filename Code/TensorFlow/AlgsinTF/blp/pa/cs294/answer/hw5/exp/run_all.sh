#!/usr/bin/env bash

##########################
### P1 Hist PointMass  ###
##########################

rm -rf data/ac_PM_bc0_s8_PointMass-v0 data/ac_PM_hist_bc0.01_s8_PointMass-v0
python train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model none -s 8 --exp_name PM_bc0_s8
python train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model hist -bc 0.01 -s 8 --exp_name PM_hist_bc0.01_s8
python plot.py data/ac_PM_bc0_s8_PointMass-v0 data/ac_PM_hist_bc0.01_s8_PointMass-v0 --save_name p1

##########################
###  P2 RBF PointMass  ###
##########################

rm -rf data/ac_PM_rbf_bc0.01_s8_sig0.2_PointMass-v0
python train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model rbf -bc 0.01 -s 8 -sig 0.2 --exp_name PM_rbf_bc0.01_s8_sig0.2
python plot.py data/ac_PM_bc0_s8_PointMass-v0 data/ac_PM_rbf_bc0.01_s8_sig0.2_PointMass-v0 --save_name p2

##########################
###  P3 EX2 PointMass  ###
##########################

rm -rf data/ac_PM_ex2_s8_bc0.05_kl0.1_dlr0.001_dh8_dti1000_PointMass-v0
python train_ac_exploration_f18.py PointMass-v0 -n 100 -b 1000 -e 3 --density_model ex2 -s 8 -bc 0.05 -kl 0.1 -dlr 0.001 -dh 8 -dti 1000 --exp_name PM_ex2_s8_bc0.05_kl0.1_dlr0.001_dh8_dti1000
python plot.py data/ac_PM_bc0_s8_PointMass-v0 data/ac_PM_ex2_s8_bc0.05_kl0.1_dlr0.001_dh8_dti1000_PointMass-v0 --legend PM_bc0_s8 PM_ex2_s8_bc0.05 --save_name p3

###########################
###    P4 HalfCheetah   ###
###########################
rm -rf data/ac_HC_bc0_HalfCheetah-v2 data/ac_HC_bc0.001_kl0.1_dlr0.005_dti1000_HalfCheetah-v2 data/ac_HC_bc0.0001_kl0.1_dlr0.005_dti10000_HalfCheetah-v2
python train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model none --exp_name HC_bc0
python train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.001 -kl 0.1 -dlr 0.005 -dti 1000 --exp_name HC_bc0.001_kl0.1_dlr0.005_dti1000
python train_ac_exploration_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --density_model ex2 -bc 0.0001 -kl 0.1 -dlr 0.005 -dti 10000 --exp_name HC_bc0.0001_kl0.1_dlr0.005_dti10000
python plot.py data/ac_HC_bc0_HalfCheetah-v2 data/ac_HC_bc0.001_kl0.1_dlr0.005_dti1000_HalfCheetah-v2 data/ac_HC_bc0.0001_kl0.1_dlr0.005_dti10000_HalfCheetah-v2 --save_name p4