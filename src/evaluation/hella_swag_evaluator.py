import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from ..models import BaseModel
from ..tokenizers import BaseTokenizer


class HellaSwagEvaluator:
    def __init__(
        self,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = None
        self.device = device

        if self.device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        self.model.to(self.device)

    def get_dataset(self, split: str = "validation"):
        return load_dataset("AlekseyKorshuk/hellaswag", split=split)

    def get_completion_probability(self, context: str, ending: str):
        full_text = context + " " + ending

        context_tokens = self.tokenizer.encode(context, return_tensors="pt").to(
            self.device
        )
        full_tokens = self.tokenizer.encode(full_text, return_tensors="pt").to(
            self.device
        )

        completion_start = context_tokens.size(1)

        with torch.no_grad():
            output = self.model(full_tokens)
            logits = output.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)

        token_probs = []
        for i in range(completion_start, full_tokens.size(1)):
            expected_token = full_tokens[0, i]
            token_log_prob = log_probs[i - 1][expected_token]
            token_probs.append(token_log_prob.item())

        return np.mean(token_probs)

    def evaluate_example(self, example: dict):
        context = example["ctx"]
        endings = example["endings"]
        correct_answer = example["label"]

        ending_scores = []
        for ending in endings:
            score = self.get_completion_probability(context, ending)
            ending_scores.append(score)

        predicted_answer = np.argmax(ending_scores)

        return {
            "is_correct": predicted_answer == correct_answer,
            "predicted": predicted_answer,
            "actual": correct_answer,
            "probabilities": ending_scores,
            "context": context,
            "endings": endings,
        }

    def evaluate(self, verbose: bool = True):
        if self.dataset is None:
            self.dataset = self.get_dataset(split="validation")

        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        results = []
        correct_predictions = 0
        total_predictions = 0
        iterator = tqdm(self.dataset) if verbose else self.dataset
        for example in iterator:
            item_result = self.evaluate_example(example)
            results.append(item_result)

            if item_result["is_correct"]:
                correct_predictions += 1
            total_predictions += 1

            if verbose and total_predictions % 100 == 0:
                current_accuracy = correct_predictions / total_predictions * 100
                iterator.set_description(f"Accuracy: {current_accuracy:.2f}%")

        accuracy = (correct_predictions / total_predictions) * 100

        return {
            "accuracy": accuracy,
            "correct": correct_predictions,
            "total": total_predictions,
            "results": results,
        }
