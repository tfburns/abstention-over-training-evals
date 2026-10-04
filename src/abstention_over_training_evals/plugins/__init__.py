"""eval-framework task plugins.

Each task lives in its own subdirectory (e.g. ``unsafe_elicitation/``, ``smoke/``) so
``--extra-tasks-dir`` can target a single task. The loader imports every ``.py`` under the directory
it is given and calls each module's ``register_tasks(registry)``. Point it at one task's directory to
load that task, or at this package to load all of them. Shared base classes live in
:mod:`abstention_over_training_evals.plugins.base` and are imported by package path, not via
``--extra-tasks-dir``.
"""
