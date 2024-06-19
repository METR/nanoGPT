# evaluate the base gpt2
# n_layer=48, n_head=25, n_embd=1600
# 1558M parameters
batch_size = 2
eval_iters = 20 # use more iterations to get good estimate
eval_only = True
wandb_log = False
init_from = 'model_with_small_embeddings.pth'
# init_from = 'model_corrupted.pth'
compile = False
device='mps'