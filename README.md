# 🧪 AI Lab v2

> *Personal laboratory for AI research and experimentation*

This repository contains my implementations and experiments with various AI architectures and training techniques. I use this space to learn by building things from scratch and testing ideas.

## 📋 What's Here

This is my workspace for:

- Implementing AI architectures to better understand how they work
- Experimenting with different training approaches
- Testing ideas and learning through hands-on coding
- Keeping track of various AI-related projects

## 🔬 Current Implementations

### Model Architectures

#### GPT-2/3 Implementation
A complete GPT-2 implementation built from scratch:

- Multi-head self-attention with causal masking
- Pre-layer normalization
- Residual connections with proper scaling
- Mixed precision training support

```python
# Example configuration
config = GPT2Config(
    vocab_size=8192, n_ctx=1024, n_layers=12, 
    d_model=768, n_heads=12, d_ff=3072, dropout=0.1
)
model = GPT2Model(config)
```

#### LLaMA-1/2 Implementation
LLaMa-inspired modern transformer implementation with:

- Rotary Position Embeddings (RoPE)
- SwiGLU activation functions
- RMSNorm normalization
- No bias terms in linear layers

```python
# Basic setup
config = LlamaConfig(
    vocab_size=32000, n_ctx=2048, d_model=4096,
    n_layers=32, n_heads=32
)
model = LlamaModel(config)
```


### Training System

#### Training Pipeline
Basic training setup with:

- Automatic mixed precision for memory efficiency
- Gradient clipping
- Learning rate scheduling (warmup + cosine decay)
- Checkpointing and resume functionality

```python
# Training example
trainer = CausalLmTrainer(
    model=model, 
    tokenizer=tokenizer,
    args=TrainingConfig(
        learning_rate=5e-4,
        train_batch_size=16,
        warmup_steps=1000
    ),
    train_dataset=dataset
)
trainer.train()
```

#### Dataset Handling
- Streaming dataset processing
- Multi-processing for tokenization
- Proper sequence handling for causal language modeling

### Text Generation

#### Sampling Methods
- Temperature scaling
- Top-p (nucleus) sampling
- Greedy decoding
- Streaming output support

```python
# Generation example
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.9
)
```


## 🔄 Ongoing Work

Currently experimenting with:

- Efficient text generation (KV cache) and Grouped Query Attention
- Implementation of Text Classification architectures (BERT, RoBERTa) and training
- Implementation of Image Classification architectures (ResNet, ViT) and training
- Various training strategies

## 📚 Structure

```
src/
├── datasets/           # Dataset loaders
├── models/             # Model implementations
│   ├── gpt2/           # GPT-2 architecture
│   └── llama/          # LLaMA architecture  
├── tokenizers/         # Various tokenization methods
├── trainers/           # Training pipeline
└── utils/              # Helper functions and utilities
```

## 🔧 Usage

The code is set up to be fairly modular. You can mix and match different components:

```python
# Pick a tokenizer
tokenizer = TinyStoriesBpe8kTokenizer()

# Configure a model
config = GPT2Config(vocab_size=tokenizer.get_vocab_size(), ...)
model = GPT2Model(config)

# Set up training
trainer = CausalLmTrainer(model, tokenizer, args, dataset)
trainer.train()
```

Most of the heavy lifting is handled automatically (mixed precision, checkpointing, etc.).

## 📝 Notes

- This is primarily for learning and experimentation
- Code quality varies as some parts are more experimental
- Not optimized for production use

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*A space for learning AI by building it from scratch.*
