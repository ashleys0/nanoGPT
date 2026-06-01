"""
train.py
from: https://github.com/karpathy/nanoGPT/blob/master/train.py


This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from tqdm import tqdm, trange
import glob
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import IterableDataset, DataLoader

import tiktoken
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node

import hydra
from omegaconf import DictConfig, OmegaConf
OmegaConf.register_new_resolver("add", lambda a, b: int(a) + int(b), replace=True)

from model import GPTConfig, GPT


from util import write_param_report, log_train_loss


# EOT token id for the gpt2 tiktoken encoding (same encoder the local .npy shards were built with)
EOT = 50256


class PackedStream(IterableDataset):
    """Pull raw-text docs from a HF streaming dataset, tokenize on the fly with the gpt2
    tiktoken encoder, concatenate with an EOT separator, and emit contiguous block_size+1
    chunks as (x, y) = (chunk[:-1], chunk[1:]). HF distributes the dataset's parquet shards
    across DataLoader workers automatically, so tokenization runs in parallel."""

    def __init__(self, hf_stream, block_size, eot=EOT):
        self.hf_stream, self.block_size, self.eot = hf_stream, block_size, eot
        self.enc = tiktoken.get_encoding("gpt2")

    def __iter__(self):
        buf = []
        need = self.block_size + 1
        for doc in self.hf_stream:
            buf.append(self.eot)
            buf.extend(self.enc.encode_ordinary(doc["text"]))
            while len(buf) >= need:
                chunk = torch.tensor(buf[:need], dtype=torch.int64)
                buf = buf[self.block_size:]  # step by block_size (1-token overlap so y aligns)
                yield chunk[:-1], chunk[1:]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    ep_num = cfg.ep_num
    out_dir = cfg.out_dir
    eval_interval = cfg.eval_interval
    log_interval = cfg.log_interval
    eval_iters = cfg.eval_iters
    eval_only = cfg.eval_only
    always_save_checkpoint = cfg.always_save_checkpoint
    save_interval = cfg.save_interval
    init_from = cfg.init_from
    ckpt_load_path = cfg.ckpt_load_path
    patience = cfg.patience
    wandb_log = cfg.wandb_log
    wandb_project = cfg.wandb_project
    wandb_run_name = cfg.wandb_run_name
    wandb_group = cfg.wandb_group
    n_shards = cfg.n_shards
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    batch_size = cfg.batch_size
    block_size = cfg.block_size
    n_layer = cfg.n_layer
    n_head = cfg.n_head
    n_embd = cfg.n_embd
    mlp_hidden = cfg.mlp_hidden
    dropout = cfg.dropout
    bias = cfg.bias
    learning_rate = cfg.learning_rate
    muon_lr = cfg.muon_lr
    muon_momentum = cfg.muon_momentum
    max_iters = cfg.max_iters
    weight_decay = cfg.weight_decay
    beta1 = cfg.beta1
    beta2 = cfg.beta2
    grad_clip = cfg.grad_clip
    decay_lr = cfg.decay_lr
    warmup_iters = cfg.warmup_iters
    lr_decay_iters = cfg.lr_decay_iters
    min_lr = learning_rate / 20 if cfg.min_lr is None else cfg.min_lr
    backend = cfg.backend
    device = cfg.device
    dtype = cfg.dtype
    compile = cfg.compile

    os.makedirs(out_dir, exist_ok=True)
    config = OmegaConf.to_container(cfg, resolve=True)

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")


    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


    # data loader
    data_dir = '/data2/ash/251B/fineweb10B'    # not rly used later (only meta.pkl lookup below)

    # --- train: stream fineweb-edu (350BT sample) and tokenize on the fly ---
    # streaming=True downloads/tokenizes just-in-time; nothing is cached to disk.
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-350BT",
                      split="train", streaming=True)
    fw = fw.shuffle(buffer_size=10000, seed=1337 + seed_offset)
    if ddp:
        fw = split_dataset_by_node(fw, rank=ddp_rank, world_size=ddp_world_size)
    train_ds = PackedStream(fw, block_size, EOT)
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=8, pin_memory=(device_type == 'cuda'),
                              prefetch_factor=4, persistent_workers=True)
    train_iter = iter(train_loader)
    print("streaming train data from HuggingFaceFW/fineweb-edu (sample-350BT)")

    # --- val: keep the local pre-tokenized shard (random access, stable/cheap eval) ---
    valset_path = '/data2/ash/251B/fineweb10B/edufineweb_val_000000.npy'
    val_data = np.load(valset_path)
    print(f"val tokens: {len(val_data):,}")

    def get_batch(split):
        if split == 'train':
            # DataLoader collates next(train_iter) into (batch_size, block_size) int64 tensors
            x, y = next(train_iter)
        else:
            ix = torch.randint(len(val_data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


    # setup for calculating "epoch" (but not technically epoch bc nanogpt is weird)
    # streaming has no fixed epoch; use a nominal 350BT so the epoch print stays meaningful
    tokens_per_step = batch_size * gradient_accumulation_steps * block_size * (ddp_world_size if ddp else 1)
    tokens_per_epoch = 350_000_000_000   # nominal: size of the sample-350BT pool
    steps_per_epoch = tokens_per_epoch / tokens_per_step
    print(f"tokens per step: {tokens_per_step:,}")
    print(f"tokens per epoch: {tokens_per_epoch:,}")
    print(f"steps per epoch: {steps_per_epoch:.1f}")


    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                    bias=bias, vocab_size=None, dropout=dropout, mlp_hidden=mlp_hidden) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        # if meta_vocab_size is None:
        #     print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = 50257  # meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        # load WEIGHTS ONLY from a checkpoint and start a fresh training run
        # (fresh optimizer + LR schedule from iter 0). Output still goes to out_dir.
        # Optimizer state is intentionally NOT restored — that would be exact resume.
        assert ckpt_load_path is not None, "init_from=resume requires ckpt_load_path"
        assert os.path.exists(ckpt_load_path), f"ckpt_load_path not found: {ckpt_load_path}"
        print(f"Loading weights from {ckpt_load_path} (fresh training, output -> {out_dir})")
        checkpoint = torch.load(ckpt_load_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise the weights won't load
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        # iter_num and best_val_loss intentionally left at their fresh defaults (0, 1e9)
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)


    # only print/save on the master process if you're using DDP
    if (not ddp) or (ddp_rank == 0):
        # write_param_report(model, f'ep{ep_num}_n_params.txt')
        pass






    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # diagnostic: snapshot one Muon-managed hidden weight and one AdamW-managed weight to track drift
    diag_muon_param = model.transformer.h[0].attn.c_q.weight
    diag_adamw_param = model.lm_head.weight
    diag_muon_init = diag_muon_param.detach().clone()
    diag_adamw_init = diag_adamw_param.detach().clone()

    # optimizers: [AdamW (embedding/lm_head), Muon (hidden 2D weights)]
    optimizers = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type,
                                            muon_lr=muon_lr, muon_momentum=muon_momentum)
    # record each param group's base LR so we can apply a multiplicative schedule
    for opt in optimizers:
        for pg in opt.param_groups:
            pg['base_lr'] = pg['lr']
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup) — returns a multiplier in [min_lr/learning_rate, 1]
    def get_lr_mult(it):
        if it < warmup_iters:
            return (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr / learning_rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        floor = min_lr / learning_rate
        return floor + coeff * (1.0 - floor)

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name,
                   group=wandb_group, config=config)






    #########################
    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    evals_since_improve = 0
    early_stop = False
    best_step = 0
    best_train_loss = float('nan')
    last_grad_norm = float('nan')

    for iter_num in trange(max_iters):

        # determine and set the learning rate for this iteration (multiplier applied to each group's base_lr)
        lr_mult = get_lr_mult(iter_num) if decay_lr else 1.0
        for opt in optimizers:
            for pg in opt.param_groups:
                pg['lr'] = lr_mult * pg['base_lr']
        lr = lr_mult * learning_rate  # for logging

        # Muon vs AdamW drift diagnostic — confirms Muon is actually moving hidden weights
        if iter_num % 500 == 0 and master_process:
            with torch.no_grad():
                muon_drift = (diag_muon_param - diag_muon_init).norm().item() / diag_muon_init.norm().item()
                adamw_drift = (diag_adamw_param - diag_adamw_init).norm().item() / diag_adamw_init.norm().item()
                adamw_lr = optimizers[0].param_groups[0]['lr']
                muon_lr_now = optimizers[1].param_groups[0]['lr']
                tqdm.write(f"[diag iter {iter_num}] adamw_lr={adamw_lr:.2e} muon_lr={muon_lr_now:.2e} "
                           f"adamw_drift={adamw_drift:.4f} muon_drift={muon_drift:.4f}")
                if wandb_log:
                    wandb.log({
                        "diag/muon_drift": muon_drift,
                        "diag/adamw_drift": adamw_drift,
                        "lr/adamw": adamw_lr,
                        "lr/muon": muon_lr_now,
                    }, step=iter_num)

        improved = False
        # unified eval: drives wandb logging, best-checkpoint save, and early-stop patience
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            epoch = iter_num / steps_per_epoch
            tqdm.write(f'step {iter_num} (epoch {epoch:.3f}): '
                       f"train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "muon_lr": muon_lr,
                    # "mfu": running_mfu * 100,
                    "grad_norm": last_grad_norm,
                }, step=iter_num)

            improved = losses['val'] < best_val_loss
            if improved:
                best_val_loss = losses['val']
                best_step = iter_num
                best_train_loss = losses['train'].item()
                evals_since_improve = 0
                # if wandb_log:
                #     wandb.run.summary['best_step'] = best_step
                #     wandb.run.summary['best_val_loss'] = best_val_loss.item()
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizers': [opt.state_dict() for opt in optimizers],
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    torch.save(checkpoint, os.path.join(out_dir, f'ep{ep_num}_best.pt'))
            else:
                evals_since_improve += 1
                if evals_since_improve >= patience:
                    tqdm.write(f"early stop at iter {iter_num} "
                               f"(no val improvement in {patience} evals)")
                    early_stop = True


        # periodic checkpoint (independent of eval cadence)
        if save_interval and iter_num > 0 and iter_num % save_interval == 0 and master_process:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizers': [opt.state_dict() for opt in optimizers],
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

        if early_stop:
            break
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # unscale, measure grad norm, and clip (measure even when clipping is off)
        for opt in optimizers:
            scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip if grad_clip != 0.0 else float('inf'),
        )
        last_grad_norm = grad_norm.item()

        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            tqdm.write(f"iter {iter_num}: loss {lossf:.4f}")
            log_train_loss(ep_num, iter_num, lossf, out_dir)

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/lossf": lossf,
                    "lr": lr,
                    "muon_lr": muon_lr,
                    "grad_norm": last_grad_norm,
                }, step=iter_num)

        # step each optimizer (scaler is a no-op for bfloat16/float32)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        model.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1


        local_iter_num += 1


    if ddp:
        destroy_process_group()

    if master_process:
        tsv_path = 'results_grid.tsv'
        write_header = not os.path.exists(tsv_path)
        best_val = best_val_loss.item() if hasattr(best_val_loss, 'item') else float(best_val_loss)
        with open(tsv_path, 'a') as f:
            if write_header:
                f.write('ep_num\tlearning_rate\tmuon_lr\ttrain_loss\tval_loss\tbest_step\n')
            f.write(f'{ep_num}\t{learning_rate}\t{muon_lr}\t{best_train_loss:.4f}\t{best_val:.4f}\t{best_step}\n')

    if wandb_log and master_process:
        wandb.finish()



if __name__ == '__main__':
    main()