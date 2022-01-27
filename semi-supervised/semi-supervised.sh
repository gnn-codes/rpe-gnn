python -u main.py --dataset 'cora' --lr 0.2 --hidden_layers 8 --hidden_dim 32 --train_iter 200 --test_iter 1 --use_saved_model True  --l_range 4 --n_sample 3 --device 'cuda:0' 

python -u main.py --dataset 'citeseer' --lr 0.2 --hidden_layers 8 --hidden_dim 32 --train_iter 200 --test_iter 1 --use_saved_model True  --l_range 4 --n_sample 4 --device 'cuda:0' 

python -u main.py --dataset 'pubmed' --lr 0.2 --hidden_layers 8 --hidden_dim 32 --train_iter 200 --test_iter 1 --use_saved_model True  --l_range 3 --n_sample 4 --device 'cuda:0' 

python -u main.py --dataset 'coauthorcs' --lr 0.2 --hidden_layers 8 --hidden_dim 32 --train_iter 200 --test_iter 1 --use_saved_model True  --l_range 2 --n_sample 4 --device 'cuda:0' 




