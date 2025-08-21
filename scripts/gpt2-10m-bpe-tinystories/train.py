import torchinfo

from src.datasets import TinyStoriesDataset
from src.models.gpt2 import GPT2Config, GPT2Model
from src.tokenizers import TinyStoriesBpe8kTokenizer
from src.trainers import CausalLmTrainer, TrainingConfig


SEQ_LEN = 256
D_MODEL = 384
N_LAYERS = 6

tokenizer = TinyStoriesBpe8kTokenizer(
    max_length=SEQ_LEN,
    padding=True,
    return_overflowing_tokens=True,
)

model_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_ctx=SEQ_LEN,
    n_layers=N_LAYERS,
    d_model=D_MODEL,
    n_heads=D_MODEL // 64,
    d_ff=D_MODEL * 2,
    dropout=0.1,
    eos_token_id=tokenizer.eos_token_id,
)

training_config = TrainingConfig(
    output_dir="./checkpoints/gpt2-10m-bpe-tinystories",
    num_train_epochs=100,
    learning_rate=5e-3,
    min_lr=5e-4,
    warmup_steps=500,
    total_steps=25000,
    betas=(0.9, 0.95),
    weight_decay=0.01,
    label_smoothing=0.0,
    train_batch_size=128,
    save_steps=100,
)

model = GPT2Model(model_config)
torchinfo.summary(model)

dataset = TinyStoriesDataset(split="train", tokenizer=tokenizer)

trainer = CausalLmTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_config,
    train_dataset=dataset,
)

trainer.train(resume_from_checkpoint=True)
