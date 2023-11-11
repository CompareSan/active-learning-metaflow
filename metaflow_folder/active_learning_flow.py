from load_unlabeled_pool_drivers import get_unlabeled_pool_drivers
from sampling_method import MarginMethod
from sampling_strategy import UncertaintySampling
from testing import evaluate
from training import fit
from utils import to_float

from metaflow import (
    FlowSpec,
    step,
)


class ActiveLearningFlow(FlowSpec):
    """
    ActiveLearningFlow is a Metaflow flow specification that implements an active learning loop.
    It starts by loading the unlabeled pool of data and selecting an initial training set.
    Then, it trains a neural network classifier using the initial training set and iteratively selects instances from the unlabeled pool to query for labels.
    The queried instances are added to the training set and the classifier is retrained.
    This process is repeated for a specified number of iterations.
    Finally, the flow ends and the final accuracy of the classifier on a test set is printed.
    """

    @step
    def start(self):
        """
        Load the data, select the initial training set, and generate the new pool.
        """
        import numpy as np
        import torchvision.transforms as transforms
        from custom_dataset import CustomDataset
        from torch.utils.data import DataLoader

        self.X_pool, self.y_pool = get_unlabeled_pool_drivers()
        self.X_pool, self.y_pool = np.array(self.X_pool) / 255, np.array(self.y_pool)

        print(f"Shape of the image {self.X_pool[0].shape}")
        # Label the test set
        self.n_initial = 1000
        np.random.seed(1234)
        self.test_idx = np.random.choice(
            range(len(self.y_pool)), size=self.n_initial, replace=False
        )
        self.X_test = self.X_pool[self.test_idx]
        self.y_test = self.y_pool[self.test_idx]
        self.X_pool = np.delete(self.X_pool, self.test_idx, axis=0)
        self.y_pool = np.delete(self.y_pool, self.test_idx, axis=0)

        # Label the initial train set for starting the AL iterations

        np.random.seed(1234)
        self.initial_idx = np.random.choice(
            range(len(self.y_pool)), size=self.n_initial, replace=False
        )
        self.X_initial = self.X_pool[self.initial_idx]
        self.y_initial = self.y_pool[self.initial_idx]
        self.X_pool = np.delete(self.X_pool, self.initial_idx, axis=0)
        self.y_pool = np.delete(self.y_pool, self.initial_idx, axis=0)

        self.custom_transform = transforms.Compose([transforms.ToTensor(), to_float])
        self.initial_dataset = CustomDataset(
            self.X_initial, self.y_initial, transform=self.custom_transform
        )
        self.test_set = CustomDataset(
            self.X_test, self.y_test, transform=self.custom_transform
        )
        self.pool = CustomDataset(
            self.X_pool, self.y_pool, transform=self.custom_transform
        )
        self.batch_size = 32
        self.train_loader = DataLoader(
            self.initial_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=True
        )
        self.pool_loader = DataLoader(self.pool, batch_size=1, shuffle=False)

        self.next(self.inital_training)

    @step
    def inital_training(self):
        import torch

        print("Initial Training")
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.model_trained = fit(self.train_loader, self.device)
        acc, prec, rec, f1_score = evaluate(
            self.model_trained, self.test_loader, self.device
        )
        self.accuracy = [acc]
        self.precision = [prec]
        self.recall = [rec]
        self.f1 = [f1_score]
        self.dimension_of_train = [self.train_loader.dataset.data.shape[0]]

        self.next(self.active_learning_loop)

    @step
    def active_learning_loop(self):
        import numpy as np
        from torch.utils.data import DataLoader

        print("################################")
        print("Start Active Learning iterations")
        self.n_iterations = 15  # number of iterations
        self.n_instances = 200  # number of instances to query
        for j in range(self.n_iterations):
            # Evaluate the unlabeled pool with some strategy

            self.indices, _ = UncertaintySampling(
                MarginMethod, self.device
            ).get_samples(self.model_trained, self.pool_loader, self.n_instances)
            # Move instances from pool_loader to train_loader
            new_train_data = np.concatenate(
                (
                    self.train_loader.dataset.data,
                    self.pool_loader.dataset.data[self.indices],
                ),
                axis=0,
            )
            new_train_labels = np.concatenate(
                (
                    self.train_loader.dataset.labels,
                    self.pool_loader.dataset.labels[self.indices],
                ),
                axis=0,
            )

            self.train_loader.dataset.data = new_train_data
            self.train_loader.dataset.labels = new_train_labels

            # Create new DataLoader objects with the updated train and pool datasets
            self.train_loader = DataLoader(
                self.train_loader.dataset, batch_size=self.batch_size, shuffle=True
            )

            # Update the pool_loader to remove the transferred instances
            self.pool_loader.dataset.data = np.delete(
                self.pool_loader.dataset.data, self.indices, axis=0
            )
            self.pool_loader.dataset.labels = np.delete(
                self.pool_loader.dataset.labels, self.indices, axis=0
            )

            print(f"Iteration {j}")
            print("###################")
            self.model_trained = fit(self.train_loader, self.device)
            acc, prec, rec, f1_score = evaluate(
                self.model_trained, self.test_loader, self.device
            )
            self.accuracy.append(acc)
            self.precision.append(prec)
            self.recall.append(rec)
            self.f1.append(f1_score)
            self.dimension_of_train.append(self.train_loader.dataset.data.shape[0])
            print(f"Accuracy on test {self.accuracy[-1]}")
            print(f"Precision on test {self.precision[-1]}")
            print(f"Recall on test {self.recall[-1]}")
            print(f"f1 score on test {self.f1[-1]}")
            print("###################")

        self.next(self.end)

    @step
    def end(self):
        print("Active Learning Finished")


if __name__ == "__main__":
    ActiveLearningFlow()
