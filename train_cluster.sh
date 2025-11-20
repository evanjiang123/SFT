#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --job-name=cluster-lora
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=duoduo.jiang@mail.mcgill.ca
#SBATCH --mail-type=END,FAIL

module load python/3.11
module load gcc/12.3 arrow/21.0.0
source ~/envs/llm-train/bin/activate

###############################################
# 1. Put ALL HF caches on SCRATCH (only these!)
###############################################
export HF_HOME=$SCRATCH/hf_home
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf_models
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

# DO NOT USE OFFLINE MODE (it breaks model loading)
unset HF_DATASETS_OFFLINE
unset HF_HUB_OFFLINE

###############################################
# 2. Set local scratch directories
###############################################
LOCAL_ROOT=${SLURM_TMPDIR}
LOCAL_MODEL=$LOCAL_ROOT/Qwen2.5-7B-Instruct
LOCAL_CLUSTER_FILE=$LOCAL_ROOT/cluster_${CLUSTER_ID}.jsonl
LOCAL_OUTPUT=$LOCAL_ROOT/qwen_loras/cluster_${CLUSTER_ID}

mkdir -p $LOCAL_ROOT/qwen_loras

echo "Extracting Qwen checkpoint tarball to local scratch..."
tar -xf /home/evan1/scratch/Multi_LLM_agent_trainning/.cache/huggingface/Qwen2.5-7B-Instruct.tar -C $LOCAL_ROOT
cp ${CLUSTER_FILE} $LOCAL_CLUSTER_FILE

###############################################
# 3. Run training
###############################################
cd /home/evan1/projects/def-rrabba/evan1/multi-llm-sim/agent_trainning

python -u train_persona_lora.py \
  --base-model $LOCAL_MODEL \
  --cluster-id ${CLUSTER_ID} \
  --cluster-file $LOCAL_CLUSTER_FILE \
  --output-dir $LOCAL_OUTPUT \
  --num-epochs 3

###############################################
# 4. Save results back to scratch
###############################################
RESULT_DIR=/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_loras/cluster_${CLUSTER_ID}
mkdir -p $RESULT_DIR
cp -r $LOCAL_OUTPUT/* $RESULT_DIR/
