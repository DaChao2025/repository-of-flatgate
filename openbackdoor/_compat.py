"""Third-party compatibility shims for modern dependency versions."""

import sys
import types


def _ensure_transformers_deepspeed_shim() -> None:
    """opendelta 0.3.2 imports transformers.deepspeed, removed in transformers >= 4.46."""
    if "transformers.deepspeed" in sys.modules:
        return

    mod = types.ModuleType("transformers.deepspeed")
    try:
        from transformers.integrations.deepspeed import (
            HfDeepSpeedConfig,
            HfTrainerDeepSpeedConfig,
            deepspeed_config,
            deepspeed_init,
            deepspeed_load_checkpoint,
            deepspeed_optim_sched,
            is_deepspeed_available,
            is_deepspeed_zero3_enabled,
            set_hf_deepspeed_config,
            unset_hf_deepspeed_config,
        )
    except ImportError:
        def deepspeed_config():
            return {}

        def is_deepspeed_zero3_enabled():
            return False

        def is_deepspeed_available():
            return False

        def deepspeed_init(*args, **kwargs):
            return None

        def deepspeed_load_checkpoint(*args, **kwargs):
            return None

        def deepspeed_optim_sched(*args, **kwargs):
            return None

        def set_hf_deepspeed_config(*args, **kwargs):
            return None

        def unset_hf_deepspeed_config(*args, **kwargs):
            return None

        HfDeepSpeedConfig = object
        HfTrainerDeepSpeedConfig = object

    for name, value in {
        "HfDeepSpeedConfig": HfDeepSpeedConfig,
        "HfTrainerDeepSpeedConfig": HfTrainerDeepSpeedConfig,
        "deepspeed_config": deepspeed_config,
        "deepspeed_init": deepspeed_init,
        "deepspeed_load_checkpoint": deepspeed_load_checkpoint,
        "deepspeed_optim_sched": deepspeed_optim_sched,
        "is_deepspeed_available": is_deepspeed_available,
        "is_deepspeed_zero3_enabled": is_deepspeed_zero3_enabled,
        "set_hf_deepspeed_config": set_hf_deepspeed_config,
        "unset_hf_deepspeed_config": unset_hf_deepspeed_config,
    }.items():
        setattr(mod, name, value)

    sys.modules["transformers.deepspeed"] = mod


_ensure_transformers_deepspeed_shim()
