import os
import torch
from lightning.pytorch.profilers import SimpleProfiler, PassThroughProfiler
from contextlib import contextmanager
from pytorch_lightning.utilities import rank_zero_only


class InferenceProfiler(SimpleProfiler):
    """
    This profiler records duration of actions with cuda.synchronize()
    Use this in test time.
    """

    def __init__(self):
        super().__init__()
        self.start = rank_zero_only(self.start)
        self.stop = rank_zero_only(self.stop)
        self.summary = rank_zero_only(self.summary)

    @contextmanager
    def profile(self, action_name: str) -> None:
        try:
            torch.cuda.synchronize()
            self.start(action_name)
            yield action_name
        finally:
            torch.cuda.synchronize()
            self.stop(action_name)


def build_profiler(args):
    name = args.profiler_name
    if name == "inference":
        return InferenceProfiler()
    elif name == "pytorch":
        dirpath = os.path.join(args.save_dir, args.exp_name)
        filename = f"{name}_profiler.txt"
        from pytorch_lightning.profilers import PyTorchProfiler

        return PyTorchProfiler(
            dirpath=dirpath,
            filename=filename,
            use_cuda=True,
            profile_memory=True,
            row_limit=100,
        )
    elif name is None:
        return PassThroughProfiler()
    else:
        raise ValueError(f"Invalid profiler: {name}")
