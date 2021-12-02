import torch
import copy
import time

class _CapsOptimizer(torch.optim.Optimizer):

    def __init__(self, params, unchange_rate=99.0,
                log_mode=False, log_dir=None):

        super(self.__class__, self).__init__(params)
        self.params = params
        self.unchange_rate = unchange_rate
        self.debug_trigger = False
        self.steps = 0;
        self.prev_params = None
        self.num_tensors = len(self.params[0]['params'])
        self.origin_grad = []
        self.log_mode = log_mode
        self.log_dir = log_dir
        self.param_size = 0
        self.skip_count = 0

        for idx in range(self.num_tensors):
            t = self.params[0]['params'][idx].data
            self.param_size += torch.numel(t)
            self.origin_grad.append(self.params[0]['params'][idx].requires_grad)


    def step(self, closure=None):
        if self.log_mode == True:
            if self.steps == 0:
                for idx in range(self.num_tensors):
                    current_size = self.params[0]['params'][idx].data.size()
                    if self.log_mode == True:
                        with open(self.log_dir + "/layer" + str(idx) + ".log", "a") as f:
                            f.write(str(current_size) + "\n")

        if self.steps != 0:
            for idx in range(self.num_tensors):
                #
                # TODO: determine concrete metric for check parameter change.
                #       How many we can torelate floating point?
                # TODO: implement more concrete and faster way to search tensor.
                #

                current_params = torch.round(self.params[0]['params'][idx].data * 10000)
                previous = torch.round(self.prev_params[0]['params'][idx].data * 10000)

                compare = torch.eq(current_params, previous)
                result = torch.count_nonzero(compare).item()
                percent = 100 * result / torch.numel(compare)

                if percent >= self.unchange_rate:
                    self.params[0]['params'][idx].requires_grad = False
                    if self.log_mode == True:
                        self.skip_count += 1
                else:
                    self.params[0]['params'][idx].requires_grad = True

                if self.log_mode == True:
                    with open(self.log_dir + "/layer" + str(idx) + ".log", "a") as f:
                        f.write(str(percent) + "\n")

        self.prev_params = copy.deepcopy(self.params)
        self.steps += 1
        return super(self.__class__, self).step(closure)


    def get_skip_count(self):
        return self.skip_count


def CapsOptimizer(optimizer, unchange_rate=90.0, log_mode=False, log_dir=None):
    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                dict(_CapsOptimizer.__dict__))
    return cls(optimizer.param_groups, unchange_rate=unchange_rate, log_mode=log_mode, log_dir=log_dir)
