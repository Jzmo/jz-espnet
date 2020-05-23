import kenlm
import torch

from espnet.nets.scorer_interface import ScorerInterface
from espnet.nets.scorer_interface import PartialScorerInterface

from abc import ABC


class Ngrambase(ABC):
    """Ngram base implemented throught ScorerInterface."""

    def __init__(self, ngram_model, token_list):
        """Initialize Ngrambase.

        Args:
            ngram_model: ngram model path
            token_list: token list from dict or model.json
 
        """
        self.chardict = [x if x != "<eos>" else "</s>" for x in token_list]
        self.charlen = len(self.chardict)
        self.lm = kenlm.LanguageModel(ngram_model)
        self.tmpkenlmstate = kenlm.State()

    def init_state(self, x):
        """Initialize tmp state."""
        state = kenlm.State()
        # since there is no <s> only </s>
        self.lm.NullContextWrite(state)
        return [0.0, state]

    def select_state(self, state, i):
        """Empty select state for scorer interface."""
        return state

    def score_partial_(self, y, next_token, state, x):
        """Score interface for both full and partial scorer.

        Args:
            y: previous char
            next_token: next token need to be score
            state: previous state
            x: encoded feature

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
 
        """

        out_state = kenlm.State()
        state[0] += self.lm.BaseScore(state[1], self.chardict[y[-1]], out_state)
        scores = torch.full(next_token.size(), state[0])
        for i, j in enumerate(next_token):
            scores[i] += self.lm.BaseScore(
                out_state, self.chardict[j], self.tmpkenlmstate
            )
        return scores, [state[0], out_state]


class NgramFullScorer(Ngrambase, ScorerInterface):
    """Fullscorer for ngram."""

    def score(self, y, state, x):
        return self.score_partial_(y, torch.tensor(range(len(self.chardict))), state, x)


class NgramPartScorer(Ngrambase, PartialScorerInterface):
    """Partialscorer for ngram."""

    def score_partial(self, y, next_token, state, x):
        return self.score_partial_(y, next_token, state, x)
