import torch
import copy
import time
import random
from collections import deque

class _CapsOptimizer(torch.optim.Optimizer):

    def __init__(self, params,
                 unchange_rate=99.0, adjust_rate=1.0,
                 lower_bound=5.0, max_bound=50.0,
                 scheduling_freq=1000, history_length=10,
                 round_factor=-1, random_select=-1,
                 hbs_length=-1,
                 custom_metric=None,
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
        self.round_factor = round_factor
        self._rf = 10 ** self.round_factor

        # Optional tensor comparison
        self.random_select = random_select
        self.hbs_length = hbs_length

        # unchange_rate scheduling
        self.adjust_rate = adjust_rate
        self.scheduling_freq = scheduling_freq
        self.lower_bound = lower_bound
        self.max_bound = max_bound
        self.metric_queue = deque(maxlen=history_length)
        self.custom_metric = custom_metric

        for idx in range(self.num_tensors):
            t = self.params[0]['params'][idx].data
            self.param_size += torch.numel(t)
            self.origin_grad.append(self.params[0]['params'][idx].requires_grad)

        if self.random_select > 0:
            self.index_list = {}


    def step(self, closure=None):
        if self.log_mode == True:
            if self.steps == 0:
                for idx in range(self.num_tensors):
                    current_size = self.params[0]['params'][idx].data.size()
                    if self.log_mode == True:
                        with open(self.log_dir + "/layer" + str(idx) + ".log", "a") as f:
                            f.write(str(current_size) + "\n")

        if self.steps != 0:
            if self.steps % self.scheduling_freq == 0:
                bad_valid = self._check_validation()
                self._schedule_unchange_rate(bad_valid)

            for idx in range(self.num_tensors):
                if self.round_factor >= 0:
                    current_params = torch.round(self.params[0]['params'][idx].data * self._rf)
                    previous = torch.round(self.prev_params[0]['params'][idx].data * self._rf)
                else:
                    current_params = self.params[0]['params'][idx].data
                    previous = self.prev_params[0]['params'][idx].data

                if self.random_select != -1:
                    percent = self._random_compare(current_params, previous, idx)
                elif self.hbs_length >= 0:
                    percent = self._weighted_compare(current_params, previous)
                else:
                    percent = self._naive_compare(current_params, previous)

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


    def get_validation(self, metric):
        self.metric_queue.append(metric)


    def _naive_compare(self, current_params, previous):
        compare = torch.eq(current_params, previous)
        result = torch.count_nonzero(compare).item()
        percent = 100 * result / torch.numel(compare)
        return percent


    def _random_compare(self, current_params, previous, index):
        total_num = torch.numel(current_params)
        if index not in self.index_list.keys():
            rand_idxs = random.sample(list(range(total_num)),
                                      int(total_num * self.random_select))
            self.index_list[index] = rand_idxs
        else:
            rand_idxs = self.index_list[index]
        t1 = torch.flatten(current_params)
        t2 = torch.flatten(previous)
        result = 0
        for idx in rand_idxs:
            if t1[idx] == t2[idx]:
                result += 1

        if len(rand_idxs) != 0:
            percent = 100 * result / len(rand_idxs)
        else:
            percent = 0
        return percent


    def _weighted_compare(self, current_params, previous):
        if self.hbs_length > 0:
            self.hbs_length += -1
        percent = 100 * result
        return percent


    def _check_validation(self):
        converted = list(self.metric_queue)
        if self.custom_metric == None:
            return all(x >= y for x, y in zip(converted, converted[1:]))
        else:
            return self.custom_metric(converted)


    def _schedule_unchange_rate(self, bad_valid=False):
        print("[CaPS System] Adjust unchange parameter rate")
        if bad_valid == False:
            self.unchange_rate = self.unchange_rate + self.adjust_rate
            if self.unchange_rate >= self.max_bound:
                self.unchange_rate = self.max_bound
        else:
            if self.unchange_rate <= self.lower_bound:
                self.unchange_rate = self.lower_bound
            else:
                self.unchange_rate = self.unchange_rate - self.adjust_rate
        print("[CaPS System] Current threshold:", self.unchange_rate)


def CapsOptimizer(optimizer,
                  unchange_rate=90.0, adjust_rate=1.0, lower_bound=5.0, max_bound=50.0,
                  scheduling_freq=1000, history_length=10, random_select=-1,
                  round_factor=4, hbs_length=-1, custom_metric=None,
                  log_mode=False, log_dir=None):

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
                dict(_CapsOptimizer.__dict__))

    return cls(optimizer.param_groups,
               unchange_rate=unchange_rate,
               adjust_rate=adjust_rate,
               lower_bound=lower_bound,
               max_bound=max_bound,
               scheduling_freq=scheduling_freq,
               history_length=history_length,
               round_factor=round_factor,
               random_select=random_select,
               hbs_length=hbs_length,
               custom_metric=custom_metric,
               log_mode=log_mode, log_dir=log_dir)
