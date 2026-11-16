"""Messages between the driver and the schedulers (names as in sglang/srt/managers/io_struct.py)."""
from dataclasses import dataclass, field


@dataclass
class TokenizedGenerateReqInput:
    rid: str
    input_ids: list
    max_new_tokens: int = 6
    temperature: float = 0.0


@dataclass
class BatchTokenIDOutput:
    rid: str
    output_ids: list


@dataclass
class FlushCacheReqInput:
    pass


@dataclass
class FlushCacheReqOutput:
    success: bool


@dataclass
class PauseGenerationReqInput:
    mode: str = "abort"           # "in_place" | "retract" | "abort"


@dataclass
class ContinueGenerationReqInput:
    pass


@dataclass
class RpcReqInput:
    method: str
    parameters: dict = field(default_factory=dict)


@dataclass
class RpcReqOutput:
    success: bool
    message: str
