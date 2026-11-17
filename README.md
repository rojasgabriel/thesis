# thesis

Behavior and electrophysiology analysis code for my Churchland Lab thesis.

```bash
uv sync --locked
```

## Import elsewhere

Run a script from another directory using this repo's environment:

```bash
uv run --project /path/to/thesis python my_analysis.py
```

In the script:

```python
from thesis.ephys.trials import build_trial_table
from thesis.ephys.units import fetch_unit_table
```

For notebooks, select this repo's `.venv` as the Python kernel.
