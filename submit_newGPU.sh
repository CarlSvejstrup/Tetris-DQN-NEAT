#BSUB -J test_2
#BSUB -o test_2%J.out
#BSUB -e test_2_err%J.err
#BSUB -n 4
#BSUB -q sxm2sh
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=8G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 8:00
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
# end of BSUB options

module load python3/3.11.3
module load cuda/12.1

source venv/bin/activate
python3 NNTemplate.py