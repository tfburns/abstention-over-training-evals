"""eval-framework model adapter for Olmo 3 32B Think served by a local ``vllm serve``.

Inference runs out of process. vLLM exposes an OpenAI-compatible endpoint, and eval-framework's
:class:`OpenAIModel` talks to it. No formatter is set, so the chat-completions API is used and
vLLM applies the Olmo chat template (needed for the Think models' ``<think>`` reasoning).

Serve the endpoint with the recommended settings, including the reasoning parser so the thinking
trace is captured separately from the answer::

    vllm serve allenai/Olmo-3-32B-Think --max-model-len 65536 --reasoning-parser olmo3
"""

from eval_framework.llm.openai import OpenAIModel

# Recommended sampling settings live in `olmo_settings` (framework-free) so the task plugin can
# share them without importing this client module. Re-exported here for callers of the client.
from abstention_over_training_evals.adapters.olmo_settings import (
    OLMO3_THINK_MAX_TOKENS,
    OLMO3_THINK_TEMPERATURE,
    OLMO3_THINK_TOP_P,
)

__all__ = [
    "OLMO3_THINK_MAX_TOKENS",
    "OLMO3_THINK_TEMPERATURE",
    "OLMO3_THINK_TOP_P",
    "Olmo3Think32B",
]


class Olmo3Think32B(OpenAIModel):
    """The final post-trained Olmo 3 32B Think checkpoint, via an OpenAI-compatible vLLM server.

    Defaults use the model card's recommended sampling. ``base_url`` and ``api_key`` come from
    ``--llm-args`` (for a local server, ``api_key=EMPTY``). To evaluate an earlier checkpoint, serve
    that checkpoint and override ``model_name``.
    """

    LLM_NAME = "allenai/Olmo-3-32B-Think"

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature if temperature is not None else OLMO3_THINK_TEMPERATURE,
            top_p=top_p if top_p is not None else OLMO3_THINK_TOP_P,
            api_key=api_key,
            base_url=base_url,
        )
