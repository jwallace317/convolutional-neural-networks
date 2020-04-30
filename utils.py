"""
Utility Module
"""

# import necessary modules
import torch


def get_accuracy(model, data_loader):
    """
    Get Accuracy

    This method reports the accuracy of the given model on the test set. This
    method utilizes batch processing and computes predicted labels in batches
    to compute the accuracy of feature classification provided by the model.

    Args:
        model (torch.nn.Module): the neural network model
        data_loader (torch.utils.data.DataLoader): the data loader

    Returns:
        accuracy (float): the accuracy of feature classification provided by the
        given model
    """

    count = 0
    correct = 0

    with torch.no_grad():
        for batch in data_loader:
            images = batch[0]
            targets = batch[1]

            # calculate predicted targets
            predictions = model(images)

            batch_count = len(batch[0])

            batch_correct = predictions.argmax(dim=1).eq(targets).sum().item()

            count += batch_count

            correct += batch_correct

    # calculate classification accuracy
    accuracy = correct / count

    return accuracy
