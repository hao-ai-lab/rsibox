"""TpModelWorker (sglang/srt/managers/tp_worker.py): one rank's model runner."""
from adapter.fixture.toysgl.srt.model_executor.model_runner import ModelRunner


class TpModelWorker:
    def __init__(self, server_args, tp_rank, tp_group):
        self.tp_rank = tp_rank
        self.model_runner = ModelRunner(server_args, tp_rank, tp_group)

    def forward_batch_generation(self, batch):
        hidden = self.model_runner.forward(batch)
        return self.model_runner.sample(hidden, batch)
