import torch.nn as nn
from attack import fgsm_attack
from tqdm import tqdm

# Training function: can perform either natural or adversarial training.
def train(model, device, optimizer, train_loader, num_epochs, epsilon, adversarial):

    # Train the model
    loss_fn = nn.CrossEntropyLoss()

    train_loss = 0
    correct = 0
    accuracies = []

    print("Start Training ...")

    for epoch in range(num_epochs):
        for batch_idx, (data, target) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss_clean = loss_fn(output, target)

            if adversarial == True:
                data.requires_grad = True  # Make sure to set requires_grad to True
                
                # Calculate gradients
                loss_clean.backward(retain_graph=True)

                # Collect datagrad
                data_grad = data.grad.data

                # Generate adversarial examples on the fly
                adv_data = fgsm_attack(data, epsilon, data_grad)
                output_adv = model(adv_data)
                loss_adv = loss_fn(output_adv, target)
                
                # Combine losses
                loss = 0.5 * (loss_clean + loss_adv)

            else:
              	loss = loss_clean
            
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(train_loader.dataset)
        accuracies.append(accuracy)

        print('Train Epoch {}:: Accuracy: {}/{} ({:.0f}%)'.format(
            epoch+1, correct, len(train_loader.dataset), accuracy))
        correct = 0

    print("Finished Training")
    return accuracies