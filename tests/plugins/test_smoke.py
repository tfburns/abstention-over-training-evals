import pytest

pytest.importorskip("eval_framework", reason="requires the `eval` extra (eval-framework[openai])")

from eval_framework.tasks.registry import Registry  # noqa: E402

from abstention_over_training_evals.plugins.smoke.smoke import (  # noqa: E402
    SafetyPipelineSmokeTest,
    register_tasks,
)


def test_register_tasks_adds_the_smoke_task():
    registry = Registry()
    register_tasks(registry)
    assert "SafetyPipelineSmokeTest" in registry


def test_smoke_task_loads_its_benign_prompt():
    task = SafetyPipelineSmokeTest.with_overwrite(num_fewshot=0, custom_subjects=None, custom_hf_revision=None)
    samples = list(task.iterate_samples())
    assert len(samples) == 1
    assert samples[0].ground_truth is None
