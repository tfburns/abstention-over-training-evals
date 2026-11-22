"""eval-framework model adapter for an Olmo 3 32B checkpoint loaded in-process with transformers.

The multiple-choice tasks score log-likelihoods of fixed completions, which eval-framework's
OpenAI-compatible client cannot get from a vLLM server, so these runs load the weights directly. This
is eval-framework's ``HFLLM`` with four changes for the purpose:

- the weights are a Hub *revision* (an intermediate checkpoint is a branch of its stage repo), fetched
  by :mod:`abstention_over_training_evals.adapters.olmo_weights` so the runner can delete them after;
- they load in bfloat16 (eval-framework leaves the dtype to torch's default, float32, which does not
  fit a 32B model on one 80 GB GPU);
- the prompt is plain text (``ConcatFormatter``, no chat template) for every checkpoint, so the base
  and post-trained stages are scored on the same prompt;
- single-token completions (the letters of the lettered tasks) are scored from one forward pass per
  prompt, batched, instead of one pass per prompt-and-letter; see :func:`fast_single_token_logprobs`.
  Anything else (the cloze option texts) takes eval-framework's path unchanged.

Used from the CLI as ``--llm-name abstention_over_training_evals.adapters.olmo_hf.OlmoHFCheckpoint
--llm-args checkpoint=base revision=stage1-step72000``, or in-process by
``scripts/nonexistent_entities_mcq/run_checkpoints.py``.
"""

from __future__ import annotations

import gc
import logging
import math
from pathlib import Path

import torch
from eval_framework.contract import Sample
from eval_framework.llm.huggingface import BaseHFLLM
from eval_framework.shared.types import ConcatCompression, PerTokenScores, RawLoglikelihood
from template_formatting.formatter import ConcatFormatter
from transformers import AutoModelForCausalLM

from abstention_over_training_evals.adapters.checkpoints import OlmoCheckpoint
from abstention_over_training_evals.adapters.olmo_weights import checkpoint_label, download_weights, parse_checkpoint

logger = logging.getLogger(__name__)


def single_token_ids(llm: BaseHFLLM, prompt: str, completions: list[str]) -> list[int] | None:
    """The token id of each completion when every one is a single token that the joint text keeps.

    eval-framework scores ``prompt + completion`` tokenised together. The shortcut is valid only when
    that joint tokenisation is the prompt's tokens followed by exactly one token per completion, which
    holds for `` A``, `` B``, ... after ``Answer:``. Returns ``None`` when it does not hold, so the
    caller falls back to the per-completion path.
    """
    prompt_ids = llm.tokenizer.encode(prompt, add_special_tokens=False)
    ids: list[int] = []
    for completion in completions:
        joint = llm.tokenizer.encode(f"{prompt}{completion}", add_special_tokens=False)
        if len(joint) != len(prompt_ids) + 1 or joint[: len(prompt_ids)] != prompt_ids:
            return None
        # The reference scores the last len(encode(completion)) joint tokens; keep that count at one.
        if len(llm.tokenizer.encode(completion, add_special_tokens=False)) != 1:
            return None
        ids.append(joint[-1])
    return ids


def fast_single_token_logprobs(llm: BaseHFLLM, samples: list[Sample], batch_size: int = 8) -> list[RawLoglikelihood]:
    """Score every sample whose completions are single tokens from one batched forward pass per prompt.

    Produces the same :class:`RawLoglikelihood` records as ``BaseHFLLM.logprobs`` (summed
    log-likelihood, token count and per-token bits per completion); samples whose completions are not
    single tokens, or whose prompt exceeds the context, go through ``BaseHFLLM.logprobs`` one by one.
    """
    prompts = [llm._formatter.format(sample.messages, output_mode="string") for sample in samples]
    ids_per_sample = [
        single_token_ids(llm, prompt, sample.possible_completions or []) for prompt, sample in zip(prompts, samples)
    ]
    max_tokens = min(filter(None, [llm.SEQ_LENGTH, llm.seq_length]))
    prompt_ids = [llm.tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
    fast = [
        i
        for i, (ids, p_ids) in enumerate(zip(ids_per_sample, prompt_ids))
        if ids is not None and len(p_ids) + 1 <= max_tokens
    ]
    results: dict[int, RawLoglikelihood] = {}
    pad_id = llm.tokenizer.pad_token_id if llm.tokenizer.pad_token_id is not None else llm.tokenizer.eos_token_id

    for start in range(0, len(fast), batch_size):
        batch = fast[start : start + batch_size]
        lengths = [len(prompt_ids[i]) for i in batch]
        width = max(lengths)
        # Right padding: with a causal mask the real tokens never see the pads that follow them.
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, i in enumerate(batch):
            input_ids[row, : lengths[row]] = torch.tensor(prompt_ids[i], dtype=torch.long)
            attention[row, : lengths[row]] = 1
        with torch.no_grad():
            out = llm.model(
                input_ids=input_ids.to(llm.device), attention_mask=attention.to(llm.device), use_cache=False
            )
            rows = torch.arange(len(batch), device=out.logits.device)
            last_positions = torch.tensor(lengths, device=out.logits.device) - 1
            last = out.logits[rows, last_positions, :].float()
            logp = torch.log_softmax(last, dim=-1)
        for row, i in enumerate(batch):
            sample, completions = samples[i], samples[i].possible_completions or []
            token_ids = ids_per_sample[i]
            assert token_ids is not None
            lps = logp[row, torch.tensor(token_ids, device=logp.device)].tolist()
            per_token = {
                completion: PerTokenScores(
                    bits=[float(-lp / math.log(2))],
                    byte_lens=[len(llm.tokenizer.decode([token_id]).encode("utf-8"))],
                )
                for completion, lp, token_id in zip(completions, lps, token_ids)
            }
            results[i] = RawLoglikelihood(
                prompt=prompts[i],
                prompt_num_tokens=lengths[row],
                concat_compression=ConcatCompression.calculate(
                    sample.messages, count_tokens=llm.count_tokens, choices=sample.possible_completions
                ),
                loglikelihoods={completion: float(lp) for completion, lp in zip(completions, lps)},
                loglikelihoods_num_tokens={completion: 1 for completion in completions},
                loglikelihoods_per_token=per_token,
                raw_loglikelihood_error=None,
            )

    slow = [i for i in range(len(samples)) if i not in results]
    if slow:
        for i, record in zip(slow, BaseHFLLM.logprobs(llm, [samples[i] for i in slow])):
            results[i] = record
    return [results[i] for i in range(len(samples))]


class OlmoHFCheckpoint(BaseHFLLM):
    """An Olmo 3 32B checkpoint (any stage, any Hub revision) scored in-process in bfloat16.

    Args:
        checkpoint: A spec for :func:`parse_checkpoint` (``base``, ``think@step_100``, a catalogued Hub
            id) or an :class:`OlmoCheckpoint`.
        revision: Overrides the spec's revision when given.
        dtype: A torch dtype name; ``bfloat16`` by default.
        weights_dir: Where to download to; defaults to ``$AOTE_WEIGHTS_DIR``.
        fast_letters: Score single-token completions with :func:`fast_single_token_logprobs` (default);
            ``False`` uses eval-framework's one-pass-per-completion path for everything.
        letter_batch_size: Prompts per forward pass on the fast path.
        bytes_per_token: eval-framework's bits-per-byte scaling override; unused by the accuracy metrics.
    """

    DEFAULT_FORMATTER = ConcatFormatter

    def __init__(
        self,
        checkpoint: str | OlmoCheckpoint,
        revision: str | None = None,
        dtype: str = "bfloat16",
        weights_dir: str | None = None,
        fast_letters: bool = True,
        letter_batch_size: int = 8,
        bytes_per_token: float | None = None,
    ) -> None:
        resolved = parse_checkpoint(checkpoint) if isinstance(checkpoint, str) else checkpoint
        self.checkpoint = resolved.at_revision(revision) if revision else resolved
        self.fast_letters = fast_letters
        self.letter_batch_size = letter_batch_size
        self.LLM_NAME = str(download_weights(self.checkpoint, Path(weights_dir) if weights_dir else None))

        # Mirrors BaseHFLLM.__init__ (eval-framework 0.14), with the dtype passed through.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = self._load_tokenizer()
        # from_pretrained returns the model in eval mode.
        self.model = AutoModelForCausalLM.from_pretrained(self.LLM_NAME, device_map="auto", dtype=getattr(torch, dtype))
        logger.info(f"Loaded {checkpoint_label(self.checkpoint)} from {self.LLM_NAME} in {dtype}")
        self._set_formatter(None)
        if bytes_per_token is not None and bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        self.bytes_per_token_scalar = (
            4.0 / bytes_per_token if bytes_per_token is not None else 4.0 / self.BYTES_PER_TOKEN
        )

    @property
    def name(self) -> str:
        return f"{type(self).__name__}_{checkpoint_label(self.checkpoint)}"

    def __del__(self) -> None:
        # eval-framework's ResponseGenerator calls ``llm.__del__`` when it is garbage-collected at the
        # end of every ``main()`` call, which would free the model between the tasks the runner scores
        # on one load. Freeing is explicit here: call :meth:`release`.
        pass

    def release(self) -> None:
        """Free the model's GPU memory (what ``BaseHFLLM.__del__`` does, on request)."""
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        if not self.fast_letters:
            return super().logprobs(samples)
        return fast_single_token_logprobs(self, samples, batch_size=self.letter_batch_size)
