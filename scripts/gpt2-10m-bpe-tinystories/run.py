import argparse

import torchinfo

from src.models.gpt2 import GPT2Config, GPT2Model
from src.utils import TextGenStdoutStreamer
from src.tokenizers import TinyStoriesBpe8kTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--checkpoint", type=str, required=True, help="Path to the checkpoint folder"
)

args = parser.parse_args()

tokenizer = TinyStoriesBpe8kTokenizer()
model = GPT2Model.from_pretrained(args.checkpoint)

streamer = TextGenStdoutStreamer(tokenizer)

torchinfo.summary(model)
while True:
    prompt = input("\n>>> ")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")[0, :-1]
    output = model.generate(input_ids, streamer=streamer)
