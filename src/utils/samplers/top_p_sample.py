import torch
import torch.nn.functional as F


def top_p_sample(logits, p=0.9):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    mask = cum_probs > p
    mask[1:] = mask[:-1].clone()
    mask[0] = False

    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum()

    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices[sampled]
