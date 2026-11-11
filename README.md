# NexoraLM: Build a Language Model from Scratch

NexoraLM is an educational GPT-style Large Language Model developed entirely from scratch using Python, PyTorch, and NumPy.

The primary objective is to build every component of a modern generative language model without relying on external pretrained models, weights, or high-level wrappers.

---

## 🏛️ Architecture Overview

NexoraLM is a **Decoder-Only Transformer** language model (similar to GPT-2) constructed completely from scratch:

```
                   Input Token IDs: (B, T)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Token Embedding]                [Positional Embedding]
    (V, embed_dim)                   (context_len, embed_dim)
            └────────────────┬────────────────┘
                             ▼  Elementwise Sum + Dropout: (B, T, embed_dim)
               ┌─────────────┴─────────────┐
               │    Transformer Blocks     │
               │         (x N)             │
               │                           │
               │   ┌───────────────────┐   │
               └──►│ Layer Normalization 1│   │
                   └─────────┬─────────┘   │
                             ▼             │
                   [Causal Multi-Head]     │
                   [  Self-Attention ]     │
                             │             │
                             ▼             │
                      [Residual Add] ◄─────┤ (Skip connection 1)
                             │             │
                             ▼             │
                   ┌───────────────────┐   │
                   │ Layer Normalization 2│   │
                   └─────────┬─────────┘   │
                             ▼             │
                   [Position-wise FFN]     │
                   (Linear->GELU->Linear)  │
                             │             │
                             ▼             │
                      [Residual Add] ◄─────┘ (Skip connection 2)
                             │
                             ▼ (Repeated across N blocks)
                     [Final LayerNorm]
                             │  Activations: (B, T, embed_dim)
                             ▼
                  [Language Modeling Head]
                  (Linear projection to V)
                  [Shared Weights with TokEmb]
                             │
                      Logits: (B, T, V)
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
          (If Targets Given)      (Next-Token Sampling)
         [Cross-Entropy Loss]    [Autoregressive Generation]
         (with ignore_index)
```

---

## ☁️ Google Colab GPU Training & Scaling (Phase 6)

NexoraLM includes a standalone, self-contained Google Colab training notebook located at [`notebooks/training.ipynb`](file:///C:/Users/admin/.gemini/antigravity/scratch/NexoraLM/notebooks/training.ipynb).

### Notebook Highlights & Features
1. **Google Drive Integration**: Mounts `/content/drive/MyDrive/NexoraLM_Colab` to store `checkpoints/` and `outputs/`. Prevents losing trained weights or logs when a Colab session disconnects.
2. **GPU & Accelerator Detection**:
   - Inspects PyTorch version, CUDA version, device name, and total VRAM.
   - Tailored for free-tier **NVIDIA T4** (15-16 GB VRAM) as well as A100/V100/L4 or CPU fallback.
3. **Automatic Mixed Precision (AMP)**:
   - Utilizes `torch.amp.autocast(device_type="cuda")` and `torch.amp.GradScaler("cuda")` to accelerate training on Tensor Cores while cutting VRAM footprint in half.
4. **Source Code Loading**:
   - **Option A (Recommended)**: Interactive field to clone your GitHub repository (`!git clone <URL>`).
   - **Option B**: Alternative to upload and extract `NexoraLM.zip`.
5. **Memory Safeguards**:
   - VRAM memory audit (`memory_allocated`, `memory_reserved`).
   - Dynamic batch size selection ($32$ or $64$ on GPU, $8$ on CPU) with `torch.cuda.empty_cache()` cleanup.
6. **Two-Stage Training Process**:
   - **Stage 1 (Smoke Test)**: 1-epoch verification on 5 validation batches to test backward pass and file writes before initiating full training.
   - **Stage 2 (Full Training)**: Full 10-epoch run with metric logging and loss curve visualization.
7. **Session Resumption**:
   - Dedicated cell to load `checkpoints/latest.pt`, restore model parameters, AdamW momentum buffers, and cosine warmup scheduler step counts.
8. **Checkpoint & Artifact Export**:
   - Exports `best.pt`, `vocabulary.json`, and `model_config.json` into an export bundle ready for deployment and evaluation.

---

## 🚂 Local Training Pipeline

### Training Commands

#### 1. Full Dataset Training
```powershell
.venv\Scripts\python.exe training/train.py --epochs 10 --batch_size 16 --lr 3e-4
```

#### 2. Quick Smoke Test
```powershell
.venv\Scripts\python.exe training/train.py --smoke_test
```

#### 3. Resume from Checkpoint
```powershell
.venv\Scripts\python.exe training/train.py --resume checkpoints/latest.pt --epochs 15
```

---

## ⚙️ Model Dimensions & Parameter Counts

NexoraLM supports arbitrary configurations derived dynamically from the tokenizer.

### Default Experimental Configuration
- **Vocabulary Size ($V$)**: 69 (derived from `vocabulary.json`)
- **Embedding Dimension ($d_{\text{model}}$)**: 256
- **Transformer Layers ($N$)**: 4
- **Attention Heads ($h$)**: 4 ($d_k = 256 / 4 = 64$)
- **Context Length ($T_{\text{max}}$)**: 256
- **FFN Hidden Dimension ($d_{\text{ff}}$)**: $4 \times 256 = 1024$
- **Dropout Rate**: 0.1
- **Weight Tying**: Enabled (`lm_head.weight == token_embeddings.weight`)
- **Total Trainable Parameters**: **3,242,752 (~3.24M)**
  *(Untied configuration: 3,260,416 parameters)*

### Small Test Configuration
- **Dimensions**: `embed_dim=128`, `num_heads=4`, `num_layers=2`, `context_length=64`
- **Total Parameters**: **413,824 (~414K)**

---

## ⚖️ Weight Tying & Weight Initialization

1. **Weight Tying Decision**:
   - Following GPT-2 and Press & Wolf (2017) *"Using the Output Embedding to Improve Language Models"*, the input token embedding matrix ($V \times d_{\text{model}}$) and the final LM head projection ($d_{\text{model}} \to V$) share the exact same weight matrix.
   - **Benefits**: Enforces semantic symmetry between token representation and output prediction, accelerates convergence, and eliminates parameter duplication.
   - **Toggleable**: Controlled via `config.tie_weights` (default `True`).

2. **Initialization Strategy**:
   - **Linear Projections**: $\mathcal{N}(0, 0.02)$, biases initialized to $0$.
   - **Embeddings**: $\mathcal{N}(0, 0.02)$, with padding token row zeroed out.
   - **LayerNorms**: Weights $= 1.0$, biases $= 0$.
   - **Residual Projections** (`out_projection` and `fc2`): Scaled by $\frac{0.02}{\sqrt{2 \times \text{num\_layers}}}$ to prevent variance accumulation through residual streams.

---

## 🔤 Custom Tokenizer & Dataset Source

- **Corpus**: *The Complete Works of William Shakespeare* (TinyShakespeare extract).
- **Licensing**: Public Domain (Project Gutenberg). Legally usable for training and educational purposes.
- **Raw File**: `data/raw/tinyshakespeare.txt` (1,115,394 bytes / ~1.11 MB).
- **Processed Split**: 90% Training (`data/processed/train.txt`, 1,003,837 chars), 10% Validation (`data/processed/val.txt`, 111,538 chars) via non-leaking sequential split.
- **Tokenizer**: Custom character-level tokenizer (`tokenizer/tokenizer.py`) with special tokens (`<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`) and dynamic vocabulary serialization (`tokenizer/vocabulary.json`).

---

## 🧪 Running Tests

Execute pytest using the project virtual environment:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 55 unit tests verify:
- **Environment & Config**: Config initialization, dynamic tokenizer binding, tensor ops.
- **Tokenizer**: Special tokens, dynamic vocab generation, encode/decode roundtrip, `<unk>` handling, persistence.
- **Preprocessing & Dataset**: Unicode NFKC normalization, sequential splitting without leakage, $y_t = x_{t+1}$ target sequence shifting, DataLoader batch shapes.
- **Transformer Components**: Token and positional embedding addition, scaled dot-product attention, causal mask enforcement ($j > i \implies 0$), multi-head concatenation, Pre-LN decoder block residual connections.
- **Complete GPT Model (`test_gpt_model.py`)**: Forward pass shapes, loss alignment, padding token exclusion, context checks, gradient backpropagation, weight tying, causal isolation.
- **Training Pipeline (`test_trainer.py`, `test_checkpoint.py`, `test_metrics.py`)**:
  - One-epoch step execution and weight updates.
  - Cosine warmup scheduler step-by-step curve.
  - Checkpoint saving and full state restoration (`model`, `optimizer`, `scheduler`, configs).
  - Best-model checkpoint protection from inferior validation epochs.
  - Overfit test on synthetic dataset verifying monotonic loss decrease.
  - Metrics logging to CSV and Matplotlib loss curve image generation.
  - Automatic CPU/CUDA device management.
- **Colab GPU Readiness (`test_colab_readiness.py`)**:
  - Mixed precision (AMP) and `GradScaler` initialization on CPU/CUDA.
  - Seamless cross-device checkpoint portability (mapping CUDA tensors onto CPU without errors).
  - Accurate scheduler state restoration and optimizer learning rate synchronization.

---

## 🗂️ Project Directory Structure

```text
NexoraLM/
├── config/
│   ├── __init__.py
│   └── config.py               # Model, Training, and Inference configurations
├── data/
│   ├── __init__.py
│   ├── raw/
│   │   └── tinyshakespeare.txt # Public domain raw corpus
│   ├── processed/
│   │   ├── train.txt           # Cleaned training corpus (1,003,837 chars)
│   │   └── val.txt             # Cleaned validation corpus (111,538 chars)
│   ├── dataset.py              # PyTorch TextDataset and DataLoader
│   └── preprocessing.py        # Text cleaning, normalization, splitting
├── tokenizer/
│   ├── __init__.py
│   ├── tokenizer.py            # CharacterTokenizer from scratch
│   └── vocabulary.json         # Persisted 69-token vocabulary
├── model/
│   ├── __init__.py
│   ├── embeddings.py           # Token & Positional Embeddings
│   ├── attention.py            # Causal Multi-Head Attention & Scaled Dot-Product
│   ├── transformer_block.py    # Pre-LN Decoder Block & FeedForwardNetwork
│   └── gpt_model.py            # Complete GPTLanguageModel
├── training/
│   ├── __init__.py
│   ├── train.py                # Main training entry point CLI
│   ├── trainer.py              # Trainer with AMP, GradScaler, and CosineWarmupScheduler
│   ├── optimizer.py            # Optimizer helpers
│   ├── checkpoint.py           # CheckpointManager (latest.pt & best.pt)
│   └── metrics.py              # MetricsTracker (CSV & loss curve plot)
├── outputs/                    # Output logs, metrics.csv, and loss_curve.png
├── checkpoints/                # Model checkpoints (latest.pt & best.pt)
├── inference/
│   ├── __init__.py
│   └── generate.py
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py
├── notebooks/
│   └── training.ipynb          # End-to-end Google Colab GPU training notebook
├── tests/
│   ├── test_environment.py     # Environment tests
│   ├── test_tokenizer.py       # Tokenizer and dataset test suite
│   ├── test_embeddings.py      # Embedding tests
│   ├── test_attention.py       # Attention and causal masking tests
│   ├── test_transformer_block.py # Decoder block and FFN tests
│   ├── test_gpt_model.py       # Full GPT model test suite
│   ├── test_trainer.py         # Trainer & overfit tests
│   ├── test_checkpoint.py      # CheckpointManager tests
│   ├── test_metrics.py         # MetricsTracker & plotting tests
│   └── test_colab_readiness.py # Colab & AMP readiness tests
├── app/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Development Roadmap

- [x] **Phase 1: Project setup and environment configuration**
- [x] **Phase 2: Dataset preparation and tokenizer implementation**
- [x] **Phase 3: Transformer components implementation**
- [x] **Phase 4: GPT model integration**
- [x] **Phase 5: Training pipeline and verification**
- [x] **Phase 6: Google Colab GPU training & scaling**
- [ ] **Phase 7: Model evaluation and text generation**
- [ ] **Phase 8: Chatbot interface**
- [ ] **Phase 9: Deployment and documentation**
#   N e x o r a L M  
 