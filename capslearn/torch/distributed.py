import os
import torch
import torch.distributed as dist
#import torch.nn.modules as Module

from datetime import timedelta
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
        self.rank = self.total_process_group.rank()
        assert(self.total_process_group.size() > 0)

        self.world_size = self.total_process_group.size()
        self.temp_group = None


        # Leanring parameters initialize
        self.modules_buffers = [list(self.module.buffers())]
        named_parameters = list(self.module.named_parameters())
        self._parameter_names = {v.__hash__(): k for k, v in sorted(named_parameters)}
        self._tensor_list = [tensor for _, tensor in sorted(named_parameters)]
        self.updatable_layers = []

        for _ in range(self.world_size):
            checker = torch.ones(len(self._tensor_list), dtype=torch.int8, device='cpu')
            self.updatable_layers.append(checker)


    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()
        return self.module(*inputs, **kwargs)


    def _sync_params(self):
        if self.total_process_group.size() == 1:
            return
        if self.broadcast_buffers:
            for idx, param in enumerate(self._tensor_list):
                if param.requires_grad == False:
                    continue

                #selective_ranks, non_ranks = self._check_ranks(idx)
                #dist.barrier(group=self.total_process_group)
                #selective_ranks = [0, 1]
                #temp_group = dist.new_group(ranks=selective_ranks)
                #self._reduce_parameters(param.detach(), group=temp_group)
                #dist.barrier(group=self.total_process_group)
                #dist.destroy_process_group(group=temp_group)

                if self.rank != 0:
                    send_req = dist.isend(param.detach().cpu(), dst=0, group=self.total_process_group)
                    send_req.wait()
                else:
                    with torch.no_grad():
                        for rank in range(self.world_size):
                            if rank != 0:
                                recv_buf = torch.zeros(param.size())
                                recv_req = dist.irecv(recv_buf, src=rank)
                                recv_req.wait()
                                param = param + recv_buf.cuda()

                        param /= self.world_size

                dist.barrier(group=self.total_process_group)
                param.requires_grad_()

            for param in self._tensor_list:
                self._broadcast_parameters(param.detach())
                param.requires_grad_()


    def _broadcast_parameters(self, param):
        dist.broadcast(param, src=0, group=self.total_process_group)


    def _reduce_parameters(self, param, target=0, group=None):
        if group != None:
            dist.reduce(param, dst=target, group=group)
        else:
            dist.reduce(param, dst=target, group=sel.total_process_group)


    def _reduce_parameter_v2(self, param, src=None, dst=0):
        if src == self.rank:
            dist.send(tensor=param, dst=dst)
        elif dst == self.rank:
            dist.recv(tensor=param, src=src)


    def _check_ranks(self, idx):
        reduce_target = []
        non_target = []
        for rank in range(self.world_size):
            if self.updatable_layers[rank][idx] == 1:
                reduce_target.append(rank)
            else:
                non_target.append(rank)
        return reduce_target, non_target
