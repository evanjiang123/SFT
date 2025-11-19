#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=cluster-lora
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=duoduo.jiang@mail.mcgill.ca
#SBATCH --mail-type=END,FAIL

module load python/3.11
module load gcc/12.3 arrow/21.0.0
source ~/envs/llm-train/bin/activate

export HF_HOME=/home/evan1/scratch/Multi_LLM_agent_trainning/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

LOCAL_ROOT=${SLURM_TMPDIR}
LOCAL_MODEL=$LOCAL_ROOT/Qwen2.5-7B-Instruct
LOCAL_CLUSTER_FILE=$LOCAL_ROOT/cluster_${CLUSTER_ID}.jsonl
LOCAL_OUTPUT=$LOCAL_ROOT/qwen_loras/cluster_${CLUSTER_ID}
mkdir -p $LOCAL_ROOT/qwen_loras

echo "Copying Qwen checkpoint to local scratch..."
mkdir -p $LOCAL_ROOT
cp -r /home/evan1/scratch/Multi_LLM_agent_trainning/.cache/huggingface/Qwen2.5-7B-Instruct $LOCAL_MODEL
cp ${CLUSTER_FILE} $LOCAL_CLUSTER_FILE

cd /home/evan1/projects/def-rrabba/evan1/multi-llm-sim/agent_trainning
accelerate launch train_persona_lora.py \
  --base-model $LOCAL_MODEL \
  --cluster-id ${CLUSTER_ID} \
  --cluster-file $LOCAL_CLUSTER_FILE \
  --output-dir $LOCAL_OUTPUT \
  --num-epochs 3 \
  --packing

RESULT_DIR=/home/evan1/scratch/Multi_LLM_agent_trainning/qwen_loras/cluster_${CLUSTER_ID}
mkdir -p $RESULT_DIR
cp -r $LOCAL_OUTPUT/* $RESULT_DIR/
