#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import numpy as np
from scipy.special import expit


class ReluActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1

class SoftmaxActivator(object):
    def forward(self,weighted_input):
        t = np.max(weighted_input)
        weighted_input -= t
        u = np.exp(weighted_input)
        return u/sum(u)
    
    def backward(self,output):
        return 1



class SigmoidActivator(object):
    def forward(self, weighted_input):
        return expit((weighted_input))

    def backward(self, output):
        return np.multiply(output, (1 - output))


class TanhActivator(object):
    def forward(self, weighted_input):
        result = 2 * expit(2 * weighted_input) -1
        return result

    def backward(self, output):
        result = 1 - np.multiply(output, output)
        return result