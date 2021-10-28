# FB15k
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
--data_path data/FB15k-betae -n 128 -b 512 -d 800 -g 30 --data FB \
-lr 0.00005 --max_steps 450001 --cpu_num 4 --valid_steps 60000 --test_batch_size 5 \
--seed 0 --drop 0.05 --tag train


# FB15k-237
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
--data_path data/FB15k-237-betae -n 128 -b 512 -d 800 -g 30 --data FB237 \
-lr 0.00005 --max_steps 300001 --cpu_num 4 --valid_steps 30000 --test_batch_size 4 \
--seed 0 --drop 0.1 --tag train


# NELL
CUDA_VISIBLE_DEVICES=0 python main.py --cuda --do_train --do_test \
--data_path data/NELL-betae -n 128 -b 512 -d 800 -g 20 --data NELL \
-lr 0.0001 --max_steps 300001 --cpu_num 4 --valid_steps 60000 --test_batch_size 4 \
--seed 0 --drop 0.2 --tag train
 