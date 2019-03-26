work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u main.py --config $work_path/config.yaml \
 --load-path=$work_path/ckpt_best.pth.tar \
 --evaluate 2>&1 | tee eval_log.txt
