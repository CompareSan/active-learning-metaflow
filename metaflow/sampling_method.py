import abc
from abc import ABC
import torch
import numpy as np
import torch.nn.functional as F

class Method(ABC):
    """
    Abstract Base Class for a method
    """
    def __init__(self, prob_dist):
        self._prob_dist = prob_dist
        
    @property
    def prob_dist(self):
        return self._prob_dist
    
    @abc.abstractmethod
    def get_score(self, *args, **kwargs):
        raise NotImplementedError(
            "Implement the `get_score`"
        )
class MarginMethod(Method):
    def __init__(self, prob_dist):
        super().__init__(prob_dist)
        self._prob_dist = prob_dist
    
    def get_score(self, sorted=False):
        """ 
        Returns the uncertainty score of a probability distribution using
        margin of confidence sampling in a 0-1 range where 1 is the most uncertain
        
        Assumes probability distribution is a pytorch tensor, like: 
            tensor([0.0321, 0.6439, 0.0871, 0.2369])
            
        Keyword arguments:
            prob_dist -- a pytorch tensor of real numbers between 0 and 1 that total to 1.0
            sorted -- if the probability distribution is pre-sorted from largest to smallest
        """
        if not sorted:
            self._prob_dist, _ = torch.sort(self._prob_dist, descending=True) # sort probs so largest is first
        
        self._prob_dist = self._prob_dist.reshape(-1)
        difference = (self._prob_dist.data[0] - self._prob_dist.data[1]) # difference between top two props
        margin_conf = 1 - difference 
        
        return margin_conf.item()