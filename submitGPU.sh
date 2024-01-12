#BSUB -J test_1
#BSUB -o test_1%J.out
#BSUB -e test_1_err%J.err
#BSUB -n 1
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 1:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

module load python3/3.11.3
module load cuda/12.1

source tetris_temp_env/bin/activate
python3 NNTemplate.py
