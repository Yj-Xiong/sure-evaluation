"""ViSQOL v3.3.3 speech-mode scoring node."""

from sure_eval.evaluation.nodes.scoring.visqol.node import (
    NODE_ID,
    NODE_VERSION,
    score_visqol,
)

__all__ = ["NODE_ID", "NODE_VERSION", "score_visqol"]
