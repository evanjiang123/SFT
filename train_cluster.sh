#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --job-name=cluster-lora
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=duoduo.jiang@mail.mcgill.ca
#SBATCH --mail-type=END,FAIL

#############################
# 1. Load Compute Canada modules
#############################
module load python/3.11
module load gcc/12.3 arrow/21.0.0

#############################
# 2. Activate YOUR NEW ENV
#############################
source ~/envs/qwen-lora/bin/activate

#############################
# 3. Make local scratch paths
#############################
export HF_HOME=$SCRATCH/hf_home
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_models
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE

#############################
# 4. Define paths
#############################
LOCAL_ROOT=$SLURM_TMPDIR
LOCAL_MODEL=$LOCAL_ROOT/Qwen2.5-7B-Instruct
LOCAL_CLUSTER_FILE=$LOCAL_ROOT/cluster_${CLUSTER_ID}.jsonl
LOCAL_OUTPUT=$LOCAL_ROOT/qwen_loras/cluster_${CLUSTER_ID}

mkdir -p $LOCAL_ROOT/qwen_loras

#############################
# 5. Extract the Qwen checkpoint
#############################
echo "Extracting Qwen checkpoint tarball to local scratch..."
tar -xf /home/evan1/scratch/Qwen2.5-7B-Instruct.tar -C $LOCAL_ROOT

cp ${CLUSTER_FILE} $LOCAL_CLUSTER_FILE

#############################
# 6. Run LoRA training
#############################
cd /home/evan1/projects/def-rrabba/evan1/multi-llm-sim/agent_trainning

python -u train_persona_lora.py \
  --base-model $LOCAL_MODEL \
  --cluster-id ${CLUSTER_ID} \
  --cluster-file $LOCAL_CLUSTER_FILE \
  --output-dir $LOCAL_OUTPUT \
  --num-epochs 3

#############################
# 7. Copy results back to persistent scratch
#############################
RESULT_DIR=/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_loras/cluster_${CLUSTER_ID}
mkdir -p $RESULT_DIR
cp -r $LOCAL_OUTPUT/* $RESULT_DIR/

echo "FINISHED cluster ${CLUSTER_ID}"
