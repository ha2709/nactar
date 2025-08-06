# 🧠 Neuron Feature Analysis on Stheno-8B

This project adapts Jake Ward’s *monosemanticity* methodology to analyze neuron activations in the **Stheno-8B** transformer model (`Sao10K/L3-8B-Stheno-v3.2`). It extracts MLP layer activations and visualizes them to explore whether individual neurons respond to specific semantic features — a property known as **monosemanticity**.

---

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the main analysis script:

 
`python analyze_sao_stheno.py`

This will:

- Load the pre-trained Stheno-8B model

- Tokenize a text prompt

- Extract activations from the final MLP block

- Run PCA on neuron activations

- Plot the neuron activity distribution and PCA results

### Example Output

A scatterplot of MLP activations projected into 2D using PCA

A histogram showing the distribution of neuron activations