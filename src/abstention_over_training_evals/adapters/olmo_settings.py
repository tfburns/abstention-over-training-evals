"""Recommended inference settings for the Olmo 3 32B Think checkpoints.

No framework imports, so the vLLM client adapter and the task plugin can share these without pulling
each other's dependencies. Values are from the model card.
"""

OLMO3_THINK_TEMPERATURE = 0.6
OLMO3_THINK_TOP_P = 0.95
OLMO3_THINK_MAX_TOKENS = 32768
