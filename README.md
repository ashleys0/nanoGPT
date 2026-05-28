2026.05.25

files, including model.py and train.py are modified from nanoGPT and modded-nanogpt. includes improvements like muon. uses wandb for logging and hydra for configs. 

```
https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_1_short/2024-10-14_ModernArch/dabaaddd-237c-4ec9-939d-6608a9ed5e27.txt
15.2 minutes	Pad embeddings, ReLU², zero-init projections, QK-norm	10/14/24	log

Tuned learning rate & rotary embeddings		
Introduced the Muon optimizer
Muon improvements
Pad embeddings, ReLU², zero-init projections, QK-norm
```



#### results
results of muon lr grid search at `results_grid.tsv`

#### settings
98.76M params total
```
n_layer: 9
n_head: 12
n_embd: 768
d_ff: 2816
batch_size: 32
block_size: 1024
```


### example usage
These override config at `conf/config.yaml`
from scratch
```
CUDA_VISIBLE_DEVICES=0 python train.py --multirun ep_num=0 n_shards=10  max_iters=1600 learning_rate=1e-2 muon_lr=1e-3 

```

training from loaded ckpt weights:
```
CUDA_VISIBLE_DEVICES=0 python train.py --multirun ep_num=0 n_shards=10 max_iters=1600 learning_rate=3e-3 muon_lr=3e-4 init_from=resume ckpt_load_path=ckpt/ep_15/ckpt_4000.pt 
```