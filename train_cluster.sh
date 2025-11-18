#!/bin/bash
#SBATCH --account=def-rrabba
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --job-name=cluster-lora
#SBATCH --output=%x-%j.out
#SBATCH --mail-user=duoduo.jiang@mail.mcgill.ca
#SBATCH --mail-type=END,FAIL

module load python/3.11
module load gcc/12.3 arrow/21.0.0
source ~/envs/llm-train/bin/activate

export HF_HOME=$SCRATCH/Multi_LLM_agent_trainning/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

cd /home/evan1/projects/def-rrabba/evan1/multi-llm-sim/Multi_LLM_agent_trainning
accelerate launch train_persona_lora.py \
  --cluster-id ${CLUSTER_ID} \
  --cluster-file ${CLUSTER_FILE} \
  --output-dir $SCRATCH/Multi_LLM_agent_trainning/qwen_loras/cluster_${CLUSTER_ID} \
  --num-epochs 3 \
  --packing
