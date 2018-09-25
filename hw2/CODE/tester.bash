#PROBLEM 4
#RUN THE PG ALGORITHM
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na

#PROBLEM 5
#RUN InvertedPendulum
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr .001 -rtg --exp_name hc_b5000_r.001

#PROBLEM 7
#RUN LUNAR LANDER
python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005

#PROBLEM 8
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b10000_r0.005
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b30000_r0.005
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name hc_b50000_r0.005

python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b10000_r0.01
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b30000_r0.01
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.01 -rtg --nn_baseline --exp_name hc_b50000_r0.01

python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 10000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b10000_r0.02
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b30000_r0.02
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name hc_b50000_r0.02

cd test2
python plot.py data/na data/baseline data/rtg data/rtg_baseline

#PLOT THE PG ALGORITHM
#BELOW ARE THE COMMANDS BUT AFTER RUNNING THE ABOVE CODE YOU WILL NEED TO CHANGE
#THE NAMES INSIDE THE <new_file_name>
# python plot.py data/lb_rtg_na_CartPole-v0_18-09-2018_22-56-17 --value AverageReturn
# python plot.py data/lb_rtg_dna_CartPole-v0_18-09-2018_22-50-00 --value AverageReturn
# python plot.py data/lb_no_rtg_dna_CartPole-v0_18-09-2018_22-42-37 --value AverageReturn
# python plot.py data/sb_rtg_na_CartPole-v0_18-09-2018_22-40-48 --value AverageReturn
# python plot.py data/sb_rtg_dna_CartPole-v0_18-09-2018_22-38-51 --value AverageReturn
# python plot.py data/sb_no_rtg_dna_CartPole-v0_18-09-2018_22-37-01 --value AverageReturn

#PLOT InvertedPendulum
# python plot.py data/hc_b5000_r.01_InvertedPendulum-v2_18-09-2018_23-08-16 --value AverageReturn

#PLOT LunarLander
# python plot.py data/ll_b40000_r0.005_LunarLanderContinuous-v2_21-09-2018_13-13-56 --value AverageReturn

#PLOT PROBLEM 8
# python plot.py data/hc_b10000_r0.005_HalfCheetah-v2_21-09-2018_17-00-10 --value AverageReturn
# python plot.py data/hc_b30000_r0.005_HalfCheetah-v2_21-09-2018_17-21-29 --value AverageReturn
# python plot.py data/hc_b50000_r0.005_HalfCheetah-v2_21-09-2018_18-30-26 --value AverageReturn
#
# python plot.py data/hc_b10000_r0.01_HalfCheetah-v2_21-09-2018_20-16-08 --value AverageReturn
# python plot.py data/hc_b30000_r0.01_HalfCheetah-v2_21-09-2018_20-31-51 --value AverageReturn
# python plot.py data/hc_b50000_r0.01_HalfCheetah-v2_21-09-2018_21-09-24 --value AverageReturn
#
# python plot.py data/hc_b10000_r0.02_HalfCheetah-v2_21-09-2018_22-12-39 --value AverageReturn
# python plot.py data/hc_b30000_r0.02_HalfCheetah-v2_21-09-2018_22-31-52 --value AverageReturn
# python plot.py data/hc_b50000_r0.02_HalfCheetah-v2_21-09-2018_23-16-20 --value AverageReturn
