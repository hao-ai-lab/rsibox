"""Decode graphs per batch size (sglang/srt/model_executor/cuda_graph_runner.py).

A CUDA graph replays the kernels it recorded and runs no Python. Here a captured graph is a
table of the decoder layers over every hidden value: replay is a lookup, so it keeps running
the layer code that was live at capture time until `capture()` runs again."""
from adapter.fixture.toysgl.srt.layers.layernorm import MODULUS

is_capture_mode = False


def get_is_capture_mode():
    return is_capture_mode


class CudaGraphRunner:
    def __init__(self, model_runner):
        self.model_runner = model_runner
        max_bs = model_runner.server_args.get("cuda_graph_max_bs", 4)
        self.capture_bs = [bs for bs in (1, 2, 4, 8, 16) if bs <= max_bs]
        self.graphs = {}
        self.output_buffers = {}
        self.capture()

    def capture(self):
        global is_capture_mode
        is_capture_mode = True
        try:
            for bs in reversed(self.capture_bs):
                graph, output_buffers = self.capture_one_batch_size(bs)
                self.graphs[bs] = graph
                self.output_buffers[bs] = output_buffers
        finally:
            is_capture_mode = False

    def capture_one_batch_size(self, bs):
        model = self.model_runner.model
        graph = [model.layers_forward(h) for h in range(MODULUS)]
        return graph, [0] * bs

    def can_run(self, bs):
        return bool(self.graphs) and bs <= max(self.graphs)

    def replay(self, hidden_states):
        bs = min(size for size in self.graphs if size >= len(hidden_states))
        graph = self.graphs[bs]
        return [graph[h] for h in hidden_states]
