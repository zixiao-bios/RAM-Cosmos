#!/usr/bin/env bash
set -euo pipefail

# 根据实际环境修改这些路径/参数
DATASET_DIR="/inspire/ssd/project/robotsimulation/public/zixiao.wang/ReWorld_Datasets/V0.1"
DATASET_BASE="/inspire/hdd/project/robotsimulation/public/Dataset/"        # 如果视频路径是相对的，设成 videos.parquet 中基路径；否则留空或和 DATASET_DIR 相同
OUTPUT_ROOT="../dataset/precomputed_latents"    # 每个 shard 会放到 ${OUTPUT_ROOT}/shard_${i}
WINDOW_SIZE=150
VIDEO_SIZE_H=240
VIDEO_SIZE_W=240
BATCH_SIZE=2
NUM_WORKERS=4
LATENT_DTYPE="float16"
EXPERIMENT_NAME=""      # 如果用默认 checkpoint，可留空
CKPT_PATH=""            # 指向已有 checkpoint，默认可留空
S3_CREDENTIAL_PATH=""   # 如果不需要从 S3 取模型，留空即可
CONTEXT_PARALLEL_SIZE=1

NUM_SHARDS=8

mkdir -p "${OUTPUT_ROOT}"

for SHARD in $(seq 0 $((NUM_SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=${SHARD} \
  python scripts/precompute_latent_windows.py \
    --dataset-dir "${DATASET_DIR}" \
    --dataset-base "${DATASET_BASE}" \
    --window-size ${WINDOW_SIZE} \
    --video-size ${VIDEO_SIZE_H} ${VIDEO_SIZE_W} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --output-dir "${OUTPUT_ROOT}/shard_${SHARD}" \
    --latent-dtype "${LATENT_DTYPE}" \
    --experiment-name "${EXPERIMENT_NAME}" \
    --ckpt-path "${CKPT_PATH}" \
    --s3-credential-path "${S3_CREDENTIAL_PATH}" \
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
    --num-shards ${NUM_SHARDS} \
    --shard-index ${SHARD} \
    --device cuda \
    --experiment-name Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt \
    --sample-cache-path "./cache/reward_samples.json" \
    >"${OUTPUT_ROOT}/shard_${SHARD}.log" 2>&1 &
done

wait
echo "All shards finished."