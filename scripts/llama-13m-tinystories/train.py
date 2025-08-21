import torchinfo
from datasets import load_dataset

from src.datasets import CausalLmDataset
from src.models.llama import LlamaConfig, LlamaModel, LlamaTokenizer
from src.trainers import Trainer, TrainingConfig


SEQ_LEN = 256
D_MODEL = 384
N_LAYERS = 6

tokenizer = LlamaTokenizer(
    vocab_file="./data/tokenizers/llama-tokenizer-tinystories-8k.model",
    add_bos_token=True,
    add_eos_token=True,
)


model_config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    n_ctx=SEQ_LEN,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=D_MODEL // 64,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

training_config = TrainingConfig(
    output_dir="./checkpoints/llama-13m-tinystories",
    num_train_epochs=100,
    learning_rate=5e-3,
    min_lr=5e-4,
    warmup_steps=500,
    total_steps=25000,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    train_batch_size=128,
    save_steps=100,
    logging_steps=10,
)

model = LlamaModel(model_config)
torchinfo.summary(model)

raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:500000]")

dataset = CausalLmDataset(
    raw_dataset,
    tokenizer,
    text_column="text",
    packed=True,
    packed_length=SEQ_LEN + 1,
    packing_stride=1,
    add_special_tokens=False,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_config,
    train_dataset=dataset,
)

trainer.train(resume_from_checkpoint=True)
