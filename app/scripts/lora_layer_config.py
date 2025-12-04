from dataclasses import dataclass


@dataclass
class LoraParams:
    r: int
    alpha: int
    target_modules: list
    learning_rate: float
    dropout: float = 0.05


def load_lora_config(name: str) -> LoraParams:
    presets = {
        "level1": LoraParams(
            r=8,
            alpha=16,
            target_modules=["q_proj", "v_proj"],
            learning_rate=2e-4,
        ),
        "level2": LoraParams(
            r=16,
            alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            learning_rate=1e-4,
        ),
        "level3": LoraParams(
            r=16,
            alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            learning_rate=5e-5,
        ),
    }

    if name not in presets:
        raise ValueError(f"Unknown LoRA preset: {name}")
    return presets[name]
