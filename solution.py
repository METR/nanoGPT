# %%

# my initial thoughts on solving this:
# Some solutions in order of quality:
# Do nothing. This would likely get a loss around 11
# Finetune model. This might have training instabilities, maybe it would get a loss of 7?
# Retrain the model from scratch. That would get a loss around 4.5
# Retrain embeddings from scratch. That would get a loss around… 4?
# Train a vocab_size squared size adapter between token and embedding, freezing other weights
# Train an adapter from small embeddings to big embeddings. This might get a loss around 3?
# Do some sort of frequency analysis to attempt to un-shuffle the embeddings.
# Same as previous, but then analyze model’s loss to determine which embeddings are still shuffled and train an adapter or do something with them
# Do PCA on embeddings for large and small model, then try to match them together somehow (this is probably great, haven’t thought it through entirely

from model import GPT, GPTConfig
import torch
device = 'mps'
print("loading model")
large_state_dict = torch.load('large_model.pth')
model = GPT(large_state_dict['config']).to(device)
del large_state_dict["config"]
model.load_state_dict(large_state_dict)
print(model)

small_state_dict = torch.load('small_correct_model.pth')
small_model = GPT(small_state_dict['config']).to(device)
del small_state_dict["config"]
small_model.load_state_dict(small_state_dict)

large_embedding = model.transformer.wte.weight.data
small_embedding = small_model.transformer.wte.weight.data



# visualize embeddings as image
import matplotlib.pyplot as plt
import numpy as np
import seaborn

def plot_embeddings(embeddings, title):
    fig, ax = plt.subplots()
    seaborn.heatmap(embeddings.detach().cpu().numpy(), ax=ax)
    ax.set_title(title)
    plt.show()

print("showing first 200 tokens of embeddings")
plot_embeddings(large_embedding[:200], "Large model embeddings")
plot_embeddings(small_embedding[:200], "small model embeddings")

print("showing first 100 hidden dimensions of embeddings")
plot_embeddings(large_embedding[:, :100], "Large model embeddings")
plot_embeddings(small_embedding[:, :100], "small model embeddings")

# PCA on small embeddings

# %%
small_svd_u, small_svd_s, small_svd_vh = torch.linalg.svd(small_embedding)

large_svd_u, large_svd_s, large_svd_vh = torch.linalg.svd(large_embedding)
# %%
def normalize(a, target):
    return a / torch.linalg.norm(a) * torch.linalg.norm(target)

print(large_svd_vh.shape)
large_in_svd_basis = torch.einsum("vh,nh->vn", large_embedding,large_svd_vh)
# plot_embeddings(large_in_svd_basis[:100, :100], "large svd")

small_in_svd_basis = torch.einsum("vh,nh->vn", small_embedding,small_svd_vh)
small_in_svd_bases_padded = torch.cat([small_in_svd_basis, torch.zeros(config_args["vocab_size"], 1600-768, device=device)], 1)
# small_projected_to_large = normalize( torch.einsum("vh,hn->vn", small_in_svd_bases_padded,large_svd_vh), large_embedding)
small_projected_to_large = normalize( torch.linalg.solve(large_svd_vh.cpu(),small_in_svd_bases_padded.transpose(1,0).cpu()).to(device).transpose(1,0), large_embedding)
plot_embeddings(small_projected_to_large[:100, :100], "small projected to large")

# %%
# save model with embedding from small
model_with_small_embeddings = GPT(model.config)
model_with_small_embeddings.load_state_dict(large_state_dict)
model_with_small_embeddings.transformer.wte.weight.data = small_projected_to_large
torch.save({**model_with_small_embeddings.state_dict(), "config":model.config}, 'model_with_small_embeddings.pth')
# %%
plot_embeddings(large_embedding[:100, :100], "large raw")
plot_embeddings(small_embedding[:100, :100], "small raw")

# %%
large_orig_variance_per_dim = torch.var(large_embedding, 0) - torch.mean(torch.var(large_embedding, 0))
print(large_orig_variance_per_dim)
projected_variance_per_dim = torch.var(small_projected_to_large, 0) - torch.mean(torch.var(small_projected_to_large, 0))
print(projected_variance_per_dim)
variance_cosine_similarity = torch.nn.functional.cosine_similarity(large_orig_variance_per_dim, projected_variance_per_dim, 0)
print("variance cosine similarity", variance_cosine_similarity)

# %%
# import model as modelmodule
# import importlib
# importlib.reload(modelmodule)
import dataclasses
# reload GPTConfig
config = modelmodule.GPTConfig(**{**dataclasses.asdict(model.config),"embedding_adapter_dim": 768})
model_small_embedding = modelmodule.GPT(config)
model_small_embedding.load_state_dict(large_state_dict, strict=False)
model_small_embedding.transformer.wte.weight.data = small_embedding
torch.save({**model_small_embedding.state_dict(), "config":config}, 'model_small_embedding_adapter.pth')
# %%
