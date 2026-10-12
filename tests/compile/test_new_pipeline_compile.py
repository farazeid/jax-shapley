"""Every registered experiment's stage programs trace, lower and compile with live-logging callbacks.

Shapes are each config's own (e.g. Mastermind-222 behaviour: 20 seeds, 10,000-row buffers,
100,000 agent steps over 16 environments); nothing executes.
"""

import pytest

from new_jax_shapley.experiment import load_configs
from new_jax_shapley.main import compile_stages


@pytest.mark.compile
@pytest.mark.parametrize("name", sorted(load_configs()))
def test_registered_experiments_compile(name: str) -> None:
    _, config = load_configs()[name]
    compile_stages(config)
