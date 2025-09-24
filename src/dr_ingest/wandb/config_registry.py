from __future__ import annotations

import catalogue
from confection import registry as cfg_registry

wandb_pattern_factories = catalogue.create(
    "dr_ingest", "wandb_pattern_factories", entry_points=False
)
wandb_hooks = catalogue.create("dr_ingest", "wandb_hooks", entry_points=False)

cfg_registry.wandb_pattern_factories = wandb_pattern_factories
cfg_registry.wandb_hooks = wandb_hooks

__all__ = [
    "wandb_pattern_factories",
    "wandb_hooks",
]
