import torch

# FGSM attack code: perturb the input image using the sign of the gradients
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel
    perturbed_image = image + epsilon * sign_data_grad
    # Clamp the perturbed image to [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)
    return perturbed_image