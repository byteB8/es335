from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F


def sample_next(
    logits: torch.Tensor,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    generator: torch.Generator | None = None,
) -> int:
    """Sample next token based on decoding strategy."""
    if strategy == "greedy" or temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    probs = F.softmax(logits / temperature, dim=-1)

    if strategy == "top_k" and top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, min(top_k, len(probs)))
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(0, top_k_indices, top_k_probs)
        probs = filtered_probs / filtered_probs.sum()

    elif strategy == "top_p" and top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=0)
        mask = cumsum_probs <= top_p
        mask[0] = True  # Ensure at least one token
        top_p_indices = sorted_indices[mask]
        filtered_probs = torch.zeros_like(probs)
        filtered_probs.scatter_(0, top_p_indices, sorted_probs[mask])
        probs = filtered_probs / filtered_probs.sum()

    return int(torch.multinomial(probs, 1, generator=generator).item())


def get_top_k_probs(logits: torch.Tensor, k: int = 10, temperature: float = 1.0) -> List[Tuple[str, float]]:
    """Get top-k token probabilities."""
    probs = F.softmax(logits / temperature, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, min(k, len(probs)))
    return [(idx.item(), prob.item()) for idx, prob in zip(top_k_indices, top_k_probs)]


def generate_words(
    model: torch.nn.Module,
    stoi: Dict[str, int],
    itos: Dict[int, str],
    prompt: List[str],
    block_size: int,
    max_new_tokens: int = 20,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device | None = None,
    seed: int | None = None,
):
    """Generate words with configurable decoding strategy."""
    model.eval()
    unk = stoi.get("<unk>", 1)
    pad = stoi.get("<pad>", 0)
    context = [pad] * max(0, block_size - len(prompt)) + \
        [stoi.get(w, unk) for w in prompt][-block_size:]
    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
    out: List[str] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor(context, dtype=torch.long,
                             device=device).view(1, -1)
            logits = model(x)[0]
            ix = sample_next(logits, strategy=strategy, temperature=temperature,
                             top_k=top_k, top_p=top_p, generator=g)
            token = itos.get(ix, "<unk>")
            if token == "." or token == "<pad>":
                break
            out.append(token)
            context = context[1:] + [ix]
    return out
