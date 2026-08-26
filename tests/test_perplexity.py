import math

import torch

from recirculation.perplexity import perplexity, shifted_nll


def test_shifted_nll_scores_exactly_next_tokens() -> None:
    # Targets are input tokens 1 and 2. Put 80% probability on each target.
    input_ids = torch.tensor([[0, 1, 2]])
    probabilities = torch.tensor(
        [[[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [1 / 3, 1 / 3, 1 / 3]]]
    )
    nll, count = shifted_nll(probabilities.log(), input_ids)
    assert count == 2
    assert math.isclose(nll.item(), -2 * math.log(0.8), rel_tol=1e-6)
    assert math.isclose(perplexity(nll.item(), count), 1.25, rel_tol=1e-6)
