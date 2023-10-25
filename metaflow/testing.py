import torch
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(model, test_loader, device):
    """
    Evaluate the performance of a given model on a test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        device (torch.device): The device (CPU or GPU) on which the evaluation will be performed.

    Returns:
        tuple: A tuple containing the accuracy, precision, recall, and F1 score of the model's predictions.

    Example Usage:
        model = MyModel()
        test_loader = DataLoader(test_dataset, batch_size=32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        acc, precision, recall, f1 = test(model, test_loader, device)
        print(f"Accuracy: {acc:.2f}%")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")
    """
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        true_labels = []
        predicted_labels = []
        for sample, labels in test_loader:
            sample = sample.to(device)
            labels = labels.to(device)

            output = model(sample)

            # value, index
            _, predictions = torch.max(output, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

        acc = n_correct / n_samples
        # Calculate precision, recall, and F1 score
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels)
        return acc, precision, recall, f1