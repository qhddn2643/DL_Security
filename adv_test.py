import torch
import torch.nn.functional as F
from attack import fgsm_attack
from tqdm import tqdm

# Generate adversarial examples on test set and evaluate.
def adv_test(model, device, test_loader, epsilon, num_epochs):
    # Accuracy counter
    correct = 0
    adv_examples = []
    final_accuracies = []

    # Test the model
    print("Start Adversarial Examples Testing ...")

    for epoch in range(num_epochs):
        # Loop over all examples in test set
        for batch_idx, (data, target) in tqdm(enumerate(test_loader), desc=f"ε={epsilon}"):

            # Send the data and label to the device
            data, target = data.to(device), target.to(device)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            # Compare for each element in the batch
            if not torch.all(init_pred.squeeze() == target):
                continue

            # Calculate the loss
            loss = F.cross_entropy(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, torch.sign(data_grad))

            # Re-classify the perturbed image
            output = model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # Compare predictions and targets element-wise
            correct_batch = (final_pred.squeeze() == target).sum().item()
            correct += correct_batch # Add the number of correct predictions in the batch to the total

            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 10):
                for i in range(data.size(0)):  # Iterate through the batch
                    if (final_pred[i].item() == target[i].item()):
                        adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                        adv_examples.append((adv_ex, init_pred[i].item(), final_pred[i].item()))

            else:
                # Save some adv examples for visualization later
                for i in range(data.size(0)):  # Iterate through the batch
                    if len(adv_examples) < 10 and (final_pred[i].item() != target[i].item()):
                        adv_ex = perturbed_data[i].squeeze().detach().cpu().numpy()
                        adv_examples.append((adv_ex, init_pred[i].item(), final_pred[i].item()))

        # Calculate final accuracy for this epsilon
        final_acc = 100. * correct / len(test_loader.dataset)
        final_accuracies.append(final_acc)

        print(f"Epoch {epoch + 1}: Epsilon: {epsilon}\tAccuracy: {correct}/{len(test_loader.dataset)} ({final_acc:.0f}%)")
        correct = 0

    print("Finished Adversarial Examples Testing\n")
    return final_accuracies