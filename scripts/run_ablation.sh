#!/usr/bin/env bash
# Ablation study driver (Sec. 4.4 of the paper).
#
# Runs three ablation variants against the full APT pipeline:
#   (a) no pretraining  : skip stage 1, train the finetune model from scratch
#   (b) no long-term    : disable the long-term encoder
#   (c) no semantic     : disable the semantic branch (c_{t,ij})
#
# This script only launches training/evaluation commands; metric aggregation
# into Nostri_Contenuti/risultati_apt_vs_sttran.txt is expected to be done
# manually after all runs complete.

set -euo pipefail
cd "$(dirname "$0")/.."

PRE_CONF=configs/apt_pretrain.yaml
FIN_CONF=configs/apt_finetune.yaml
MODE="${MODE:-predcls}"

run_pretrain() {
    local save_path=$1
    shift
    python -m train.train_pretrain --config "${PRE_CONF}" --set \
        mode="${MODE}" save_path="${save_path}" "$@"
}

run_finetune() {
    local pretrain_ckpt=$1
    local save_path=$2
    shift 2
    python -m train.train_finetune --config "${FIN_CONF}" --set \
        mode="${MODE}" pretrain_ckpt="${pretrain_ckpt}" \
        save_path="${save_path}" "$@"
}

run_eval() {
    local ckpt=$1
    shift
    python -m eval.eval_apt --config "${FIN_CONF}" --set \
        mode="${MODE}" pretrain_ckpt="${ckpt}" "$@"
}

echo "=============================================================="
echo "Full APT (pretrain + finetune)"
echo "=============================================================="
run_pretrain data/apt_pretrain_full/
run_finetune data/apt_pretrain_full/apt_pretrain_latest.tar data/apt_finetune_full/
run_eval data/apt_finetune_full/apt_finetune_latest.tar

echo "=============================================================="
echo "Ablation (a) — no pretraining"
echo "=============================================================="
run_finetune "" data/apt_finetune_nopre/
run_eval data/apt_finetune_nopre/apt_finetune_latest.tar

echo "=============================================================="
echo "Ablation (b) — no long-term encoder"
echo "=============================================================="
run_pretrain data/apt_pretrain_nolong/ use_long_term=false
run_finetune data/apt_pretrain_nolong/apt_pretrain_latest.tar data/apt_finetune_nolong/ use_long_term=false
run_eval data/apt_finetune_nolong/apt_finetune_latest.tar use_long_term=false

echo "=============================================================="
echo "Ablation (c) — no semantic branch"
echo "=============================================================="
run_pretrain data/apt_pretrain_nosem/ use_semantic_branch=false
run_finetune data/apt_pretrain_nosem/apt_pretrain_latest.tar data/apt_finetune_nosem/ use_semantic_branch=false
run_eval data/apt_finetune_nosem/apt_finetune_latest.tar use_semantic_branch=false
