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


    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()
        return self.module(*inputs, **kwargs)


    def _sync_params(self):
        if self.total_process_group.size() == 1:
            return

        # Synchronize whole process
        dist.barrier(group=self.total_process_group)
        self._gather_valid_param()

        if self.broadcast_buffers:
            for idx, param in enumerate(self._tensor_list):
                if self.rank != 0:
                    if param.requires_grad == True:
                        send_req = dist.isend(param.detach().cpu(), dst=0, group=self.total_process_group)
                        #send_req.wait()
                else:
                    with torch.no_grad():
                        cnt = 1
                        for rank in range(self.world_size):
                            if rank != 0:
                                if self._check_valid_param(rank, idx) == True:
                                    recv_buf = torch.zeros(param.size())
                                    recv_req = dist.irecv(recv_buf, src=rank)
                                    recv_req.wait()
                                    param = param + recv_buf.cuda()
                                    cnt += 1

                        param /= cnt

                dist.barrier(group=self.total_process_group)
                param.requires_grad_()

            for param in self._tensor_list:
                self._broadcast_parameters(param.detach())
                param.requires_grad_()


    def _broadcast_parameters(self, param):
        dist.broadcast(param, src=0, group=self.total_process_group)


    def _gather_valid_param(self):
        buf = []
        for tensor in self._tensor_list:
            if tensor.requires_grad == True:
                buf.append(True)
            else:
                buf.append(False)
        send_buf = torch.tensor(buf, device='cpu')
        dist.gather(send_buf, gather_list=self.updatable_layers, dst=0, group=self.total_process_group)


    def _check_valid_param(self, rank, param_idx):
        return True if self.updatable_layers[rank][param_idx] == True else False
