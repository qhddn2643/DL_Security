import torch
import torch.nn as nn
from tqdm import tqdm

# Testing function: can evaluate clean or adversarial examples based on epsilon, option to apply a defense (post-processing) on the input.
def test(model, device, test_loader, num_epochs, defense_fn=None, defense_params=None):

    # Test the model
    loss_fn = nn.CrossEntropyLoss()

    test_loss = 0
    correct = 0
    accuracies = []

    print("Start Testing ...")

    for epoch in tqdm(range(num_epochs)):
        with torch.no_grad():
            for batch_idx, (data, target) in tqdm(enumerate(test_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
                data, target = data.to(device), target.to(device)

                if defense_fn is not None:
                    # Apply the defense (e.g., feature squeezing or Gaussian blur)
                    if 'bit_depth' in defense_params:
                        data = defense_fn(data, **defense_params)
                    else:
                        data = defense_fn(data, device, **defense_params)

                # Reshape data for the DNN
                output = model(data)
                test_loss += loss_fn(output, target)
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability

                correct += pred.eq(target.view_as(pred)).sum().item()

            accuracy = 100. * correct / len(test_loader.dataset)
            accuracies.append(accuracy)

            print('Test Epoch {}:: Accuracy: {}/{} ({:.0f}%)'.format(
                epoch+1, correct, len(test_loader.dataset), accuracy))
            correct = 0

    print("Finished Testing")
    return accuracies