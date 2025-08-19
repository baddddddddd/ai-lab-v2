import argparse

import torchinfo

from src.models.llama import LlamaConfig, LlamaModel
from src.utils import TextGenStdoutStreamer
from src.tokenizers import TinyStoriesBpe8kTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the checkpoint folder"
)

args = parser.parse_args()

tokenizer = TinyStoriesBpe8kTokenizer()
model = LlamaModel.from_pretrained(args.checkpoint)

streamer = TextGenStdoutStreamer(tokenizer)

torchinfo.summary(model)
while True:
    prompt = input("\n>>> ")

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    output = model.generate(input_ids, streamer=streamer)
