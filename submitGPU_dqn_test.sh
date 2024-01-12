#BSUB -J dqn_run_test
#BSUB -o dqn_run_test%J.out
#BSUB -e dqn_run_test%J.err
#BSUB -n 4
#BSUB -q gpuv10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

module load python3/3.11.3
module load cuda/11.8

source tetris_temp_env/bin/activate
python3 ./DQN/src_dqn/train_dqn.py