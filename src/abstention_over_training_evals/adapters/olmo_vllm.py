"""eval-framework model adapter for Olmo 3 32B Think served by a local ``vllm serve``.

Inference runs out of process. vLLM exposes an OpenAI-compatible endpoint, and eval-framework's
:class:`OpenAIModel` talks to it. No formatter is set, so the chat-completions API is used and
vLLM applies the Olmo chat template (needed for the Think models' ``<think>`` reasoning).

Serve the endpoint with the recommended settings, including the reasoning parser so the thinking
trace is captured separately from the answer::

    vllm serve allenai/Olmo-3-32B-Think --max-model-len 65536 --reasoning-parser olmo3
"""

import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from eval_framework.llm.openai import OpenAIModel
from eval_framework.shared.types import Error, RawCompletion
from template_formatting.formatter import Message

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
    that checkpoint and override ``model_name``. ``timeout`` (seconds) raises the per-request limit for
    long reasoning traces; ``max_workers`` is the number of requests in flight at once.
    """

    LLM_NAME = "allenai/Olmo-3-32B-Think"

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_workers: int = 256,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature if temperature is not None else OLMO3_THINK_TEMPERATURE,
            top_p=top_p if top_p is not None else OLMO3_THINK_TOP_P,
            api_key=api_key,
            base_url=base_url,
        )
        if timeout is not None:
            # A long reasoning trace can take longer than the client library's default of 600 s.
            self._client = self._client.with_options(timeout=float(timeout))
        self._max_workers = int(max_workers)

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> list[RawCompletion]:
        """Generate every request in its own thread, and keep one failure to one request.

        The framework's implementation shares a thread pool sized by the CPU count (about 30
        requests in flight, far below what a vLLM server runs) and lets one failed request raise
        out of the whole batch, which the framework then records as errors for every sample in it,
        with no way to resume them. Here ``max_workers`` requests run at once and a failure becomes
        an error record for that request alone.
        """

        def one(single: Sequence[Message]) -> RawCompletion:
            try:
                return OpenAIModel.generate_from_messages(
                    self, [single], stop_sequences, max_tokens, temperature, top_p
                )[0]
            except Exception as exc:  # recorded per request; the run goes on
                return RawCompletion(
                    prompt="",
                    prompt_num_tokens=0,
                    completion="",
                    completion_num_tokens=0,
                    raw_completion_error=Error(
                        error_class=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
                    ),
                )

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            return list(executor.map(one, messages))
