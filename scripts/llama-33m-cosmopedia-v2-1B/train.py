import torchinfo
from datasets import load_dataset


from src.datasets import CausalLmDataset
from src.models.llama import LlamaConfig, LlamaModel, LlamaTokenizer
from src.trainers import Trainer, TrainingArguments
from src.utils.schedulers import CosineDecayWithLinearWarmupLR

SEQ_LEN = 512
BATCH_SIZE = 52
GRAD_ACCUM = 3

WARMUP_STEPS = 300
TOTAL_STEPS = 13000
MAX_LR = 5.5e-3
MIN_LR = 1e-5

D_MODEL = 512
N_LAYERS = D_MODEL // 64
N_HEADS = D_MODEL // 64

tokenizer = LlamaTokenizer(
    "./data/tokenizers/llama-tokenizer-16k-cosmopedia-v2-1B/tokenizer.model",
    add_bos_token=True,
    add_eos_token=True,
)

config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    n_ctx=SEQ_LEN,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

args = TrainingArguments(
    output_dir="./checkpoints/llama-33m-cosmopedia-v2-1B-base",
    overwrite_output_dir=True,
    train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=TOTAL_STEPS,
    learning_rate=MAX_LR,
    weight_decay=0.1,
    adam_beta1=0.90,
    adam_beta2=0.95,
    adam_epsilon=1e-4,
    max_grad_norm=1.0,
    warmup_steps=WARMUP_STEPS,
    logging_steps=100,
    save_steps=1000,
    fp16=True,
)

model = LlamaModel(config)

packed_dataset = load_dataset(
    "eluxeee/cosmopedia-v2-1B-packed-512-llama-tokenizer-16k", split="train"
)

dataset = CausalLmDataset(
    packed_dataset,
    tokenizer,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataloader=dataset.get_dataloader(batch_size=BATCH_SIZE),
)

scheduler = CosineDecayWithLinearWarmupLR(
    optimizer=trainer.get_optimizer(),
    warmup_steps=WARMUP_STEPS,
    total_steps=TOTAL_STEPS,
    min_lr=MIN_LR,
)

trainer.scheduler = scheduler

trainer.train(resume_from_checkpoint=False)
