from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    A custom dataset class for machine learning tasks.

    Args:
        data (list): The input data for the dataset.
        labels (list): The corresponding labels for the data.
        transform (callable, optional): An optional transformation function to preprocess the data.

    Example Usage:
        dataset = CustomDataset(data, labels, transform)
        length = len(dataset)
        sample, label = dataset[idx]
    """

    def __init__(self, data, labels, transform=None):
        """
        Initializes the CustomDataset object with the given data, labels, and an optional transformation function.

        Args:
            data (list): The input data for the dataset.
            labels (list): The corresponding labels for the data.
            transform (callable, optional): An optional transformation function to preprocess the data.
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a specific sample and label from the dataset at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the sample and its corresponding label.
        """
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label
