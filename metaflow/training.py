from model_cnn_drivers import CNN
import torch
from torch import nn
import numpy as np

def fit(train_loader, device):
    """
    Trains a convolutional neural network (CNN) model using the PyTorch library.

    Args:
        train_loader (torch.utils.data.DataLoader): A data loader that provides batches of training samples and labels.
        device (torch.device): The device (CPU or GPU) on which the model and data should be loaded.

    Returns:
        model (model_cnn_drivers.CNN): The trained CNN model.
    """

    model = CNN(num_classes=2, n_channel=1)
    model.to(device)

    NUM_EPOCHS = 10
    LEARNING_RATE = 3e-4
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    losses_per_epoch = []
    for epoch in range(NUM_EPOCHS):
        losses_per_batch = []
        for i, (sample, labels) in enumerate(train_loader):
            sample = sample.to(device)
            labels = labels.to(device)

            #forward
            output = model(sample)
            loss = criterion(output, labels)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_per_batch.append(loss.item())


        losses_per_epoch.append(np.mean(losses_per_batch))
        print(f'Epoch {epoch+1}, Train Loss: {losses_per_epoch[-1]}')

    return model