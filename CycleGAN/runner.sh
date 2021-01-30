#!/bin/bash
#!/bin/bash
#SBATCH -p batch # partition (this is the queue your job will be added to)
#SBATCH -n 2 # number of cores (here 2 cores requested)
#SBATCH --time=02:00:00 # time allocation, which has the format (D-HH:MM), here set to 1 hour
#SBATCH --gres=gpu:2 # generic resource required (here requires 1 GPU)
#SBATCH --mem=32GB # specify memory required per node (here set to 16 GB)

# Notification configuration
#SBATCH --mail-type=FAIL                                        # Send a notification email when the job fails (=FAIL) 
#SBATCH --mail-user=arpit.garg@adelaide.edu.au          # Email to which notifications will be sent 
# Execute your script

module load Anaconda3/5.0.1
module load Python/3.6.1-foss-2016b
   


#source activate lxmert
source /fast/users/a1784072/virtualenvs/trial/bin/activate

module load CUDA/9.0.176

module load cuDNN/7.0-CUDA-9.0.176



source $FASTDIR/virtualenvs/trial/bin/activate


python CycleGAN.py

deactivate

