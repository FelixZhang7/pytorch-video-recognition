work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n$2 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 \
python -u main.py --config $work_path/config.yaml \
2>&1|tee $work_path/full_log.txt
# --load-path=$work_path/ckpt.pth.tar --recover \
# 2>&1|tee $work_path/full_log.txt
#--fp16 --dynamic-loss-scale --fp16-normal-bn 2>&1|tee $work_path/full_log.txt
#--load-path=$work_path/ckpt.pth.tar --recover
