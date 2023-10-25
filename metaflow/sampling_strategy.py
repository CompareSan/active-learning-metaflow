import torch.nn.functional as F
import torch
import numpy as np
import abc
from abc import ABC

class SamplingStrategy(ABC):
    """
    Abstract Base Class for a Sampling Strategy
    """

    def __init__(self, method, device):
        self.method = method
        self.device = device
        

    @abc.abstractmethod
    def get_samples(self, *args, **kwargs):
        raise NotImplementedError(
            "Implement the `get_samples`"
        )

class UncertaintySampling(SamplingStrategy):
    def __init__(self, method, device):
        super().__init__(method, device)
        self.method = method
        self.device = device

    def get_samples(self, model, unlabeled_data, number):
        """
        Returns the indices and scores of the top samples based on the uncertainty scores.

        Args:
            model (nn.Module): The neural network model used for prediction.
            unlabeled_data (DataLoader): The dataset containing unlabeled samples.
            method (function): The uncertainty sampling method to be applied to the probability distribution.
            device (torch.device): The device (e.g., CPU or GPU) on which the model and data should be processed.
            number (int): The number of top samples to be returned.

        Returns:
            indices (np.ndarray): An array of integers representing the indices of the top samples.
            scores (np.ndarray): An array of floats representing the uncertainty scores of the top samples.
        """
        if self.method == 'random':
            indices = np.random.choice(range(unlabeled_data.dataset.data.shape[0]),
                                        size=number, replace=False)
            scores = []
        else:
            samples = []
            with torch.no_grad():
                for i, (sample, _) in enumerate(unlabeled_data):
                    sample = sample.to(self.device)
                    logits = model(sample)
                    prob_dist = F.softmax(logits, dim=1)
                    score = self.method(prob_dist).get_score()
                    samples.append([i, score])

            samples.sort(reverse=True, key=lambda x: x[1])
            samples_to_be_returned = samples[:number:]
            indices = np.array(samples_to_be_returned)[:, 0]
            scores = np.array(samples_to_be_returned)[:, 1]
        
        return indices.astype(int), scores