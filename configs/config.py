from dataclasses import dataclass, field

@dataclass
class Cnnconfig:
    input_channel: int = 1
    model_info: list[tuple[int,int]] = field(default_factory=lambda: [(8,3),(16,5),(32,7)])