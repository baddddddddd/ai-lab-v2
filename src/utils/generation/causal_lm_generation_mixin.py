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
        streamer: BaseStreamer | None = None,
    ):
        generated = input_ids.clone().tolist()
        if streamer is not None:
            streamer.put(input_ids)

        model_input = torch.tensor(
            [generated],
            dtype=torch.long,
            device=input_ids.device,
        )
        past_key_values = None
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
