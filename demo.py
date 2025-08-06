import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm

# Load model and tokenizer
model_name = "Sao10K/L3-8B-Stheno-v3.2"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
model.to(device)
model.eval()

# Inspect model to find MLP layer
print(model.transformer.h[-1].mlp)  # Final block’s MLP

# Hook to extract MLP output
activations = []
def mlp_hook(module, input, output):
    activations.append(output)

hook = model.transformer.h[-1].mlp.register_forward_hook(mlp_hook)

# Input text
text = "The quick brown fox jumps over the lazy dog. " * 4
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

# Forward pass
with torch.no_grad():
    _ = model(**inputs)

hook.remove()

# Get activations: shape (batch_size, seq_len, n_mlp_neurons)
mlp_output = activations[0].detach().cpu().squeeze(0).numpy()  # shape (seq_len, hidden_dim)

# Optional: Run PCA on MLP output
pca = PCA(n_components=2)
mlp_pca = pca.fit_transform(mlp_output)

# 🔍 Simple visualization
plt.figure(figsize=(8, 6))
plt.scatter(mlp_pca[:, 0], mlp_pca[:, 1], alpha=0.7)
plt.title("PCA of MLP Activations (Final Layer, Stheno-8B)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()

# 🔢 Optional: neuron activation histogram
plt.figure(figsize=(10, 4))
plt.hist(mlp_output.flatten(), bins=100)
plt.title("Histogram of MLP neuron activations")
plt.show()