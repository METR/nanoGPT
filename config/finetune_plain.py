import time

out_dir = 'out-openwebtext'
eval_interval = 5
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'openwebtext'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'openwebtext'
init_from = 'large_model.pth'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 8
gradient_accumulation_steps = 1
max_iters = 40

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
device='mps'
compile=False