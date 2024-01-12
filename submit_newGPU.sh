#BSUB -J changed_test1
#BSUB -o test_2%J.out
#BSUB -e test_2_err%J.err
#BSUB -n 4
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

module load python3/3.11.5
module load cuda/12.1

source tetris_temp_env/bin/activate
python3 NNTemplate.py