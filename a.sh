export CUDA_VISIBLE_DEVICES=0; python trinity_pred.py --world_size 8 --rank 1 &
export CUDA_VISIBLE_DEVICES=0; python trinity_pred.py --world_size 8 --rank 4 &
export CUDA_VISIBLE_DEVICES=0; python trinity_pred.py --world_size 8 --rank 2 &
export CUDA_VISIBLE_DEVICES=1; python trinity_pred.py --world_size 8 --rank 8 &
export CUDA_VISIBLE_DEVICES=1; python trinity_pred.py --world_size 8 --rank 5 &
export CUDA_VISIBLE_DEVICES=1; python trinity_pred.py --world_size 8 --rank 7 &
export CUDA_VISIBLE_DEVICES=2; python trinity_pred.py --world_size 8 --rank 6 &
export CUDA_VISIBLE_DEVICES=3; python trinity_pred.py --world_size 8 --rank 0 &
