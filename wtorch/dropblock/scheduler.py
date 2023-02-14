import numpy as np
from torch import nn


class LinearScheduler(nn.Module):
    def __init__(self, dropblock, start_value=-1.0, stop_value=-1.0, begin_step=0,end_step=-1):
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        if start_value>=0.0 and stop_value>=0.0 and end_step>0:
            nr_steps = end_step-begin_step
            self.drop_values = np.linspace(start=start_value, stop=stop_value, num=int(nr_steps))
            self.drop_prob = None
        elif start_value<0 and stop_value<0:
            self.drop_values = None
            self.drop_prob = self.dropblock.drop_prob
            if end_step<0:
                end_step = 1e9
        self.begin_step = begin_step
        self.end_step = end_step

    def forward(self, x):
        return self.dropblock(x)

    def step(self,step=None):
        if step is not None:
            self.i = step
        else:
            self.i += 1
        if self.i>=self.begin_step and self.i<=self.end_step:
            if self.drop_prob is not None:
                self.dropblock.drop_prob = self.drop_prob
            elif self.drop_values is not None and self.i-self.begin_step < len(self.drop_values):
                self.dropblock.drop_prob = self.drop_values[self.i-self.begin_step]
        elif self.drop_prob is not None:
            self.dropblock.drop_prob = 0.0

