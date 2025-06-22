import random
from typing import Optional, Tuple
from optimum.utils import (
    DummyInputGenerator,
    NormalizedConfig,
    DEFAULT_DUMMY_SHAPES,
)


class DummyTimestepInputGenerator(DummyInputGenerator):
    """
    Generates dummy time step inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "timestep",
        "text_embeds",
        "time_ids",
        "timestep_cond",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedConfig,
        batch_size: int = DEFAULT_DUMMY_SHAPES["batch_size"],
        random_batch_size_range: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        self.task = task
        self.vocab_size = normalized_config.vocab_size
        self.time_ids = 5
        if random_batch_size_range:
            low, high = random_batch_size_range
            self.batch_size = random.randint(low, high)
        else:
            self.batch_size = batch_size

    def generate(
        self,
        input_name: str,
        framework: str = "pt",
        int_dtype: str = "int64",
        float_dtype: str = "fp32",
    ):
        shape = [self.batch_size, 1]

        if input_name == "timestep":
            return self.random_int_tensor(
                shape, max_value=self.vocab_size, framework=framework, dtype=int_dtype
            )

        return self.random_float_tensor(
            shape, max_value=self.vocab_size, framework=framework, dtype=float_dtype
        )
