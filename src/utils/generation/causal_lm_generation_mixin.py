import torch

from ..base_streamer import BaseStreamer
from ..samplers import top_p_sample


class CausalLmGenerationMixin:
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.9,
        use_cache: bool = True,
        streamer: BaseStreamer | None = None,
    ):
        if use_cache:
            generated = input_ids.clone().tolist()
            if streamer is not None:
                streamer.put(input_ids)

            model_input = torch.tensor(
                [generated],
                dtype=torch.long,
                device=input_ids.device,
            )
            past_key_values = None
            hard_max_new_tokens = self.config.n_ctx - len(generated) + 1
            max_new_tokens = min(max_new_tokens, hard_max_new_tokens)
            start_pos = 0
            for _ in range(max_new_tokens):
                output = self.forward(
                    model_input,
                    past_key_values=past_key_values,
                    start_pos=start_pos,
                    use_cache=True,
                )
                logits = output.logits[0, -1]
                past_key_values = output.past_key_values
                start_pos = len(generated)

                if temperature > 0.0:
                    logits /= temperature
                    if top_p > 0.0:
                        next_token = top_p_sample(logits, p=top_p).item()
                    else:
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(logits).item()

                generated.append(next_token)
                model_input = torch.tensor(
                    [[next_token]], dtype=torch.long, device=input_ids.device
                )

                if streamer is not None:
                    streamer.put(torch.tensor([next_token], device=input_ids.device))

                if (
                    self.config.eos_token_id is not None
                    and next_token == self.config.eos_token_id
                ):
                    break

            if streamer is not None:
                streamer.end()

            return torch.tensor(generated, device=input_ids.device)
        else:
            # old sliding window implementation
            from collections import deque

            generated = deque(input_ids.clone().tolist(), maxlen=self.config.n_ctx)
            if streamer is not None:
                streamer.put(input_ids)

            for _ in range(max_new_tokens):
                model_input = torch.tensor(
                    [generated], dtype=torch.long, device=input_ids.device
                )
                logits = self.forward(model_input).logits
                logits = logits[0, -1]

                if temperature > 0.0:
                    logits /= temperature
                    if top_p > 0.0:
                        next_token = top_p_sample(logits, p=top_p)
                    else:
                        probs = F.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(logits).item()

                generated.append(next_token)

                if streamer is not None:
                    streamer.put(torch.tensor([next_token], device=input_ids.device))

                if (
                    self.config.eos_token_id is not None
                    and next_token == self.config.eos_token_id
                ):
                    break

            if streamer is not None:
                streamer.end()

            return torch.tensor(generated, device=input_ids.device)
