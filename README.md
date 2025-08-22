# 🧪 AI Lab v2

> *Personal laboratory for AI research and experimentation*

This repository contains my implementations and experiments with various AI architectures and training techniques. I use this space to learn by building things from scratch and testing ideas.

## 📋 What's Here

This is my workspace for:

- Implementing AI architectures to better understand how they work
- Experimenting with different training approaches and optimizations
- Testing ideas and learning through hands-on coding
- Keeping track of various AI-related projects

## 🔬 Current Implementations

### Model Architectures

#### GPT-2 Implementation
A complete GPT-2 implementation built from scratch with modern optimizations:

- **Multi-head self-attention** with causal masking and Flash Attention support
- **Pre-layer normalization** with residual connections and proper scaling
- **Mixed precision training** support with automatic gradient scaling
- **KV caching** for efficient text generation
- **Streaming generation** with configurable sampling methods

```python
# Example configuration and usage
from src.models.gpt2 import GPT2Config, GPT2Model

config = GPT2Config(
    vocab_size=8192, 
    n_ctx=1024, 
    n_layers=12, 
    d_model=768, 
    n_heads=12, 
    d_ff=3072, 
    dropout=0.1,
    eos_token_id=1
)
model = GPT2Model(config)
```

#### LLaMA Implementation
Modern transformer implementation following LLaMA architecture:

- **Rotary Position Embeddings (RoPE)** for better position encoding
- **SwiGLU activation functions** in feed-forward networks
- **RMSNorm normalization** instead of LayerNorm
- **No bias terms** in linear layers for efficiency
- **KV caching** support for fast inference

```python
# Basic LLaMA setup
from src.models.llama import LlamaConfig, LlamaModel

config = LlamaConfig(
    vocab_size=32000, 
    n_ctx=2048, 
    d_model=4096,
    n_layers=32, 
    n_heads=32,
    bos_token_id=1,
    eos_token_id=2
)
model = LlamaModel(config)
```

### Advanced Features

#### KV Cache System
Efficient caching system for fast text generation:

- **Base KV Cache** interface for extensibility
- **Static KV Cache** for fixed-size context windows
- **Layer-wise caching** with automatic memory management
- **Lazy initialization** for memory efficiency

#### Text Generation
Comprehensive generation system with multiple sampling strategies:

- **Temperature scaling** for controlling randomness
- **Top-p (nucleus) sampling** for quality control
- **Greedy decoding** for deterministic output
- **Streaming generation** with real-time output
- **Configurable stopping criteria**

```python
# Generation example with streaming
from src.utils import TextGenStdoutStreamer

streamer = TextGenStdoutStreamer(tokenizer)
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9,
    use_cache=True,
    streamer=streamer
)
```

### Tokenization System

#### Multiple Tokenizer Implementations
- **Base Tokenizer** - Abstract interface for all tokenizers
- **TinyStories BPE 8K** - Pre-trained BPE tokenizer for TinyStories dataset
- **Stripped ASCII Tokenizer** - Simple character-level tokenizer
- **LLaMA Tokenizer** - SentencePiece-based tokenizer with training capabilities

```python
# Tokenizer usage example
from src.tokenizers import TinyStoriesBpe8kTokenizer

tokenizer = TinyStoriesBpe8kTokenizer()
encoded = tokenizer.encode("Hello, world!")
decoded = tokenizer.decode(encoded)
```

### Dataset Processing

#### Flexible Dataset System
- **Base Dataset** classes with standard interfaces
- **Causal LM Dataset** for language model training
- **Streaming support** for large datasets
- **Multi-processing tokenization** for efficiency
- **Sequence packing** for optimal GPU utilization
- **Corpus dumping** utilities for data analysis

```python
# Dataset setup with packing
from src.datasets import CausalLmDataset
from datasets import load_dataset

raw_dataset = load_dataset("roneneldan/TinyStories", split="train")
dataset = CausalLmDataset(
    raw_dataset,
    tokenizer,
    packed=True,
    packed_length=1024,
    packing_stride=512
)
```

### Training System

#### Comprehensive Training Pipeline
- **Mixed precision training** with automatic gradient scaling
- **Gradient clipping** for training stability
- **Advanced learning rate scheduling** (warmup + cosine decay)
- **Checkpointing and resume** functionality with full state preservation
- **RNG state management** for reproducibility
- **Flexible logging** and monitoring

```python
# Training setup
from src.trainers import Trainer, TrainingConfig

config = TrainingConfig(
    output_dir="./checkpoints",
    num_train_epochs=3,
    learning_rate=5e-4,
    train_batch_size=16,
    warmup_steps=1000,
    save_steps=5000,
    logging_steps=100
)

trainer = Trainer(
    model=model, 
    tokenizer=tokenizer,
    args=config,
    train_dataset=dataset
)
trainer.train()
```

#### Advanced Scheduling
- **Linear Warmup + Cosine Annealing** learning rate schedule
- **Configurable warmup steps** and minimum learning rate
- **Step-based scheduling** with automatic optimization

## 📚 Repository Structure

```
src/
├── datasets/           # Dataset loaders and processing
│   ├── base_dataset.py           # Abstract dataset interfaces
│   └── causal_lm_dataset.py      # Language modeling datasets
├── models/             # Model implementations
│   ├── base_config.py            # Configuration management
│   ├── base_model.py             # Model base class
│   ├── model_output.py           # Structured model outputs
│   ├── gpt2/                     # GPT-2 implementation
│   └── llama/                    # LLaMA implementation
├── tokenizers/         # Tokenization implementations
│   ├── base_tokenizer.py         # Tokenizer interface
│   ├── tinystories_bpe_8k.py     # BPE tokenizer
│   ├── stripped_ascii_tokenizer.py # Character tokenizer
│   └── llama_tokenizer.py        # SentencePiece tokenizer
├── trainers/           # Training pipeline
│   ├── trainer.py                # Main training logic
│   ├── training_config.py        # Training configuration
│   └── schedulers/               # Learning rate schedulers
└── utils/              # Utility functions
    ├── generation/               # Text generation utilities
    ├── kv_cache/                 # KV caching system
    ├── samplers/                 # Sampling strategies
    └── base_streamer.py          # Streaming interfaces
```

## 🔧 Usage Examples

### Quick Start
```python
# Import necessary components
from src.models.gpt2 import GPT2Config, GPT2Model
from src.tokenizers import TinyStoriesBpe8kTokenizer
from src.datasets import CausalLmDataset
from src.trainers import Trainer, TrainingConfig
from datasets import load_dataset

# Set up tokenizer
tokenizer = TinyStoriesBpe8kTokenizer()

# Configure model
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_ctx=1024,
    n_layers=12,
    d_model=768,
    n_heads=12,
    d_ff=3072,
    dropout=0.1,
    eos_token_id=tokenizer.eos_token_id
)
model = GPT2Model(config)

# Prepare dataset
raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")
dataset = CausalLmDataset(raw_dataset, tokenizer, packed=True, packed_length=1024)

# Set up training
training_args = TrainingConfig(
    output_dir="./outputs",
    num_train_epochs=1,
    learning_rate=5e-4,
    train_batch_size=8,
    warmup_steps=100,
    save_steps=500
)

# Train the model
trainer = Trainer(model, tokenizer, training_args, dataset)
trainer.train()
```

### Generation with Streaming
```python
from src.utils import TextGenStdoutStreamer

# Load trained model
model = GPT2Model.from_pretrained("./outputs/checkpoint-1000")
tokenizer = TinyStoriesBpe8kTokenizer()

# Set up streaming
streamer = TextGenStdoutStreamer(tokenizer)

# Generate text
prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

generated = model.generate(
    input_ids,
    max_new_tokens=200,
    temperature=0.8,
    top_p=0.9,
    use_cache=True,
    streamer=streamer
)
```

## 📝 Notes

- This is primarily for learning and experimentation
- Code quality varies as some parts are more experimental
- Not optimized for production use

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*A space for learning AI by building it from scratch.*
