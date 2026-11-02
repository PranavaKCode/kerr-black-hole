#!/bin/bash
#SBATCH -J kerr-web
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:40:00
# headless chromium captures on a cluster node. submit from the webgl dir with a python env that has
# playwright active (sbatch exports it); partition and gpu go on the command line:
#   sbatch -p <partition> --gres=gpu:1 cluster/batch.sbatch specs.json [page]
#   sbatch -p <partition> -c 32 --mem=32G cluster/batch.sbatch specs.json dist/index.html cpu
python cluster/batch.py "$1" "${2:-dist/index.html}" $3
