import torchinfo
from datasets import load_dataset

from src.datasets import CausalLmDataset
from src.models.llama import LlamaConfig, LlamaModel
from src.tokenizers import TinyStoriesBpe8kTokenizer
from src.trainers import Trainer, TrainingConfig


SEQ_LEN = 256
D_MODEL = 384
N_LAYERS = 6

tokenizer = TinyStoriesBpe8kTokenizer()

model_config = LlamaConfig(
    vocab_size=tokenizer.get_vocab_size(),
    n_ctx=SEQ_LEN,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=D_MODEL // 64,
    bos_token_id=None,
    eos_token_id=tokenizer.eos_token_id,
)

training_config = TrainingConfig(
    output_dir="./checkpoints/llama-13m-bpe-tinystories",
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

raw_dataset = load_dataset("roneneldan/TinyStories", split="train[:200000]")


def tok_prep(text: str, **kwargs):
    text += tokenizer.eos_token
    return (text, kwargs)


tokenizer.prepare_for_tokenization = tok_prep

dataset = CausalLmDataset(
    raw_dataset,
    tokenizer,
    text_column="text",
    add_special_tokens=False,
    padding=True,
    truncation=True,
    max_length=SEQ_LEN + 1,
    stride=1,
    return_overflowing_tokens=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_config,
    train_dataset=dataset,
)

trainer.train(resume_from_checkpoint=True)
