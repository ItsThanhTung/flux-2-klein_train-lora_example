import random
import numpy as np
from typing import List

from torch.utils.data import Dataset
import os

    
class ProbPickingDataset(Dataset):
    """A dataset wrapper for picking dataset with probability."""

    def __init__(self, datasets: List[dict], probs: List[float] = None, length: int = None):
        '''
        Args:
            datasets: list of dataset objects
            probs: list of probabilities to sample for each dataset
            length: the length of the combine dataset (if not provided, use the sum of the dataset lengths)
        '''
        super().__init__()
        assert len(datasets) == len(probs), "datasets and probs must have the same length"

        self.dataset_list = []
        self.dataset_probs = []

        for dataset, prob in zip(datasets, probs):
            self.dataset_list.append(dataset)
            self.dataset_probs.append(prob)

        # Set length: use provided length, or use the weighted sum of the dataset lengths
        if length is not None:
            self._length = length
        else:
            # weighted sum
            # self._length = int(sum([len(ds) * prob for ds, prob in zip(self.dataset_list, [dataset_prob["prob"] for dataset_prob in datasets])]))
            self._length = int(sum([len(ds) for ds in self.dataset_list]))

    def __getitem__(self, idx):
        """
        Randomly select a dataset based on probability and get an item from it.
        The idx parameter is used to ensure deterministic behavior with the same seed.
        """
        # Use a combination of idx and random selection to maintain some determinism
        # while still respecting the probability distribution
        selected_dataset_idx = np.random.choice(len(self.dataset_list), p=self.dataset_probs)
        selected_dataset = self.dataset_list[selected_dataset_idx]
        data_idx = idx % len(selected_dataset)
        return selected_dataset[data_idx]

    def __len__(self):
        return self._length
