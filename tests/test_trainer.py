import math

import pytest

from app.training.trainer import DEFAULT_NBS, batch_plan


@pytest.mark.parametrize("train_count", [3, 8, 19, 40, 100, 255])
@pytest.mark.parametrize("max_batch", [4, 16, 64])
def test_small_datasets_get_several_optimizer_steps_per_epoch(train_count, max_batch):
    batch, nbs = batch_plan(train_count, max_batch)
    accumulate = max(round(nbs / batch), 1)  # how Ultralytics turns nbs into accumulation steps
    steps_per_epoch = math.ceil(train_count / batch) / accumulate
    assert 1 <= batch <= max_batch
    assert steps_per_epoch >= min(3, math.ceil(train_count / batch))


def test_twenty_diagrams_step_every_batch():
    # The case that used to train with one batch per epoch and one update every three epochs.
    assert batch_plan(19, 32) == (4, 4)


def test_large_datasets_keep_ultralytics_defaults():
    assert batch_plan(5000, 16) == (16, DEFAULT_NBS)


def test_memory_limit_wins_over_the_step_target():
    assert batch_plan(1000, 2)[0] == 2
