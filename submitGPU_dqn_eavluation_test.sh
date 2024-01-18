#BSUB -J dqn_eval_1_
#BSUB -o dqn_eval_1_%J.out
#BSUB -e dqn_eval_1_%J.err
#BSUB -n 4
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

module load python3/3.11.3
module load cuda/11.8

source tetris_temp_venv/bin/activate
python3 -u ./DQN/src_dqn/ai_run.py
