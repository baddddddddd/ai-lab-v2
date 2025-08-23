import torchinfo
from datasets import load_dataset

from src.datasets import CausalLmDataset
from src.models.llama import LlamaConfig, LlamaModel, LlamaTokenizer
from src.trainers import Trainer, TrainingConfig


SEQ_LEN = 512
BATCH_SIZE = 100

WARMUP_STEPS = 100
TOTAL_STEPS = 10000
MAX_LR = 2e-3
MIN_LR = MAX_LR * 0.1

D_MODEL = 384
N_LAYERS = D_MODEL // 64
N_HEADS = D_MODEL // 64

tokenizer = LlamaTokenizer(
    "./data/tokenizers/llama-tokenizer-10k-cosmopedia-100k/llama-tokenizer-10k-cosmopedia-100k.model",
    add_bos_token=True,
    add_eos_token=True,
)

model_config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    n_ctx=SEQ_LEN,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

training_config = TrainingConfig(
    output_dir="./checkpoints/llama-14m-cosmopedia-100k",
    num_train_epochs=10,
    learning_rate=MAX_LR,
    min_lr=MIN_LR,
    warmup_steps=WARMUP_STEPS,
    total_steps=TOTAL_STEPS,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    train_batch_size=BATCH_SIZE,
    save_steps=100,
    logging_steps=10,
)

model = LlamaModel(model_config)
torchinfo.summary(model)

raw_dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", split="train")

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
