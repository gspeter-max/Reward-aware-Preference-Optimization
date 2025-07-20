# Reward-aware Preference Optimization (RPO)

> **A powerful experimental framework combining reward modeling with causal language models and reverse KL-divergence to achieve fine-grained preference learning.**

---

## 🧠 Overview

This repository implements **Reward-aware Preference Optimization (RPO)** — a custom training strategy that aligns model behavior to learned preferences without full RL loops. The core objective is to optimize a language model using:

* A **Reward Model** trained on human-like preference data
* **Reverse KL Divergence** loss to regulate model updates
* **Quantized Transformers (4-bit)** for efficiency using `BitsAndBytes`

---

## 🔧 Components

### 1. `Reward_Model.py`

Defines a reward model architecture:

* Built on top of a pretrained causal LM (e.g. GPT-2)
* Uses a linear layer on final logits for scalar reward prediction
* Trained on pairwise preference data with a contrastive loss:

```python
loss = -logsigmoid(score_1 - score_2)
```

---

### 2. `dataset.py`

* Dynamically builds a dataset of prompt-response pairs
* Tokenizes input as:

```text
prompt : <question> [SEP] response : <response>
```

* Returns batches compatible with the model using PyTorch DataLoader.

---

### 3. `runner.py`

Full training loop:

* Loads models with 4-bit quantization
* Computes log-likelihood scores from policy and reference models
* Applies reverse KL divergence loss between reward and model distributions:

```python
loss = reward_probs * log(reward_probs / model_probs)
```

* Supports score normalization and evaluation.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/Reward-aware-Preference-Optimization.git
cd Reward-aware-Preference-Optimization
pip install -r requirements.txt
```

> Ensure CUDA is available and your GPU supports `bfloat16` for full precision compatibility.

---

## 🚀 Quick Start

```bash
python runner.py
```

Sample output:

```bash
loss is : -- tensor(0.3842, device='cuda:0', dtype=torch.float32)
```

---

## 🧪 Key Features

* ✅ 4-bit GPT-2 quantized inference using `BitsAndBytes`
* ✅ Reverse KL divergence loss
* ✅ Dynamic prompt-response formatting
* ✅ Simple reward shaping for custom training


## 📂 Files

| File               | Description                                       |
| ------------------ | ------------------------------------------------- |
| `runner.py`        | Main training pipeline                            |
| `Reward_Model.py`  | Reward network using linear output layer          |
| `dataset.py`       | Tokenizer + batch builder                         |
| `requirements.txt` | Minimal dependencies (transformers, bitsandbytes) |

---

