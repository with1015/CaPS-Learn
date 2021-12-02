import os
import torch
import torch.distributed as dist
#import torch.nn.modules as Module

from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.distributed_c10d import new_group

class DistributedDataParallel(torch.nn.Module):

    def __init__(self, module,
                 device_ids=None,
                 broadcast_buffers=True):

        super(DistributedDataParallel, self).__init__()
        # Common initialize
        self.module = module
        self.device_ids = device_ids
        self.broadcast_buffers = broadcast_buffers
        self.require_forward_param_sync = broadcast_buffers
        self.require_backward_param_sync = True

        # Communication initialize
        self.total_process_group = _get_default_group()
        self.group_list = []
        self.rank = self.process_group.rank()
        assert(self.total_process_group.size() > 0)

        self.world_size = self.process_group.size()
        self.temp_group = None

        # Leanring parameters initialize
        self.modules_buffers = [list(self.module.buffers())]
        named_parameters = list(self.module.named_parameters())
        self._parameter_names = {v.__hash__(): k for k, v in sorted(named_parameters)}
        self._tensor_list = [tensor for _, tensor in sorted(named_parameters)]


    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()
        return self.module(*inputs, **kwargs)


    def _sync_params(self):
        if self.total_process_group.size() == 1:
            return
        if self.broadcast_buffers:
            for param in self._tensor_list:
                if param.requires_grad == False:
                    group = self._create_new_group(ranks=[0,1])
                    self._reduce_parameters(param.detach(), group)
                with torch.no_grad():
                    param /= self.world_size
                param.requires_grad_()

            for param in self._tensor_list:
                self._broadcast_parameters(param.detach())
                param.requires_grad_()


    def _broadcast_parameters(self, param):
        dist.broadcast(param, src=0, group=self.total_process_group)


    def _reduce_parameters(self, param, target=0, group=self.total_process_group):
        dist.reduce(param, dst=target, group=group)


    def _create_new_group(self, ranks=[]):
        temp_group = new_group(ranks=ranks, backends='gloo')
        return temp_group
