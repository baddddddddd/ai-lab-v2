import argparse

import torchinfo

from src.models.llama import LlamaConfig, LlamaModel, LlamaTokenizer
from src.utils import TextGenStdoutStreamer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the checkpoint folder"
)

args = parser.parse_args()

tokenizer = LlamaTokenizer(
    "./data/tokenizers/llama-tokenizer-10k-cosmopedia-100k/llama-tokenizer-10k-cosmopedia-100k.model",
)

model = LlamaModel.from_pretrained(args.checkpoint)

streamer = TextGenStdoutStreamer(tokenizer)

torchinfo.summary(model)
while True:
    prompt = input("\n>>> ")

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    output = model.generate(input_ids, streamer=streamer, use_cache=True)
