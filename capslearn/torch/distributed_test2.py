import os
import torch
import torch.distributed as dist
import torch.nn.modules import Module

from torch.multiprocessing. import Process
from torch.distributed.distributed_c10d import _get_default_group

class DistributedDataParallel(Module):

    def __init__(self, module,
                 device_ids=None
                 broadcast_buffers=True):

        super(DistributedDataParallel, self).__init__()
        self.module = module
        self.device_ids = device_ids
        self.broadcast_buffers = broadcast_buffers
        self.require_forward_param_sync = broadcast_buffer
        self.require_backward_param_sync = True

        self.modules_buffers = [list(self.module.buffers())]
        named_parameters = list(self.module.named_parameters())
        self._parameter_names = {v.__hash__(): k for k, v in sorted(named_parameters)}
        self._tensor_list = [tensor for _, tensor in named_parameters]
        self.process_group = _get_default_group()


    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params():
        return self.module(*inputs, **kwargs)


    def _sync_params(self):
        with torch.no_grad():
            if self.broadcast_buffers:
                for param in self._tensor_list:
                    self._broadcast_parameters(param)


    def _broadcast_parameters(self, param):
        dist.broadcast(param, src=0, group=self.process_group)
