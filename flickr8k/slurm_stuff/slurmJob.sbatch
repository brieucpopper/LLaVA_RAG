#!/bin/bash
#SBATCH -JslurmJob                 # Job name
#SBATCH -N1 --ntasks-per-node=1 --gres=gpu:V100:1 --mem-per-gpu=16G         # Number of nodes and cores per node required
#SBATCH -t3:15:00                            # Duration of the job (Ex: 15 mins)
#SBATCH -oReport-%j.out                  # Combined output and error messages file
#SBATCH --mail-type=BEGIN,END,FAIL       # Mail preferences
#SBATCH --mail-user=bpopper3@gatech.edu # E-mail address for notifications
cd $SLURM_SUBMIT_DIR                     # Change to working directory


module load anaconda3                # Load module dependencies
conda init --reverse

conda activate vlm


export HUGGINGFACE_HUB_CACHE="/home/hice1/bpopper3/scratch/hug_face_files/hub_cache"
export HF_HOME="/home/hice1/bpopper3/scratch/hug_face_files/hf_home"

srun python compute_embeddings.py