import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Step 1: Load Stheno Model ----------
MODEL_NAME = "Sao10K/L3-8B-Stheno-v3.2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32)
model.to(device)
model.eval()

# ---------- Step 2: Hook into the MLP Layer ----------
# inspect model.transformer.h to find correct MLP layer
layer_activations = []

def get_activation_hook(name):
    def hook(module, input, output):
        # Example: capturing hidden states from MLP output
        layer_activations.append(output[0].detach().cpu())
    return hook

# Choose a layer (e.g., final MLP layer)
target_layer_idx = -1  # final transformer block
mlp_layer = model.model.layers[target_layer_idx].mlp  # Check model arch to match this

hook = mlp_layer.register_forward_hook(get_activation_hook("final_mlp"))

# ---------- Step 3: Load Text and Collect Activations ----------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:5%]")  # Small subset
texts = [item["text"] for item in dataset if len(item["text"]) > 20][:100]

all_activations = []

for text in tqdm(texts):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    layer_activations.clear()

    with torch.no_grad():
        _ = model(**inputs)

    if layer_activations:
        act = layer_activations[0].squeeze(0).numpy()  # (seq_len, hidden_dim)
        all_activations.append(act)

# Concatenate across tokens and examples
all_activations = np.concatenate(all_activations, axis=0)  # shape: (tokens, hidden_dim)

# ---------- Step 4: Neuron Resampling ----------
# Resample activations to analyze neuron-specific patterns
def resample_neuron_firings(activations, top_k=10):
    """
    Find top_k neurons that fire most consistently.
    """
    neuron_means = activations.mean(axis=0)  # (hidden_dim,)
    neuron_vars = activations.var(axis=0)    # (hidden_dim,)

    signal_to_noise = neuron_means / (np.sqrt(neuron_vars) + 1e-5)
    top_indices = np.argsort(-np.abs(signal_to_noise))[:top_k]

    return top_indices, signal_to_noise[top_indices]

top_neurons, snr_scores = resample_neuron_firings(all_activations)

print("Top neurons (by SNR):", top_neurons)
print("SNR scores:", snr_scores)

# ---------- Step 5: Clustering & Sparsity Analysis ----------
def plot_kmeans_projection(activations, neuron_ids):
    for nid in neuron_ids:
        vals = activations[:, nid].reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(vals)

        plt.hist(vals, bins=50, alpha=0.5, label=f'Neuron {nid}')
        plt.axvline(kmeans.cluster_centers_[0], color='red')
        plt.axvline(kmeans.cluster_centers_[1], color='green')
        plt.title(f"Neuron {nid} activation histogram")
        plt.legend()
        plt.show()

plot_kmeans_projection(all_activations, top_neurons[:3])

# ---------- Optional: Sparsity Metric ----------
def compute_sparsity(activations, threshold=1.0):
    """
    Compute what fraction of neurons are rarely activated.
    """
    above_thresh = np.abs(activations) > threshold
    sparsity = 1.0 - (np.sum(above_thresh) / activations.size)
    return sparsity

print("Sparsity:", compute_sparsity(all_activations))
