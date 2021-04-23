import torch as th
from WGAN.fft_functions import from_time_to_frequency, from_fft_evaluate_abs

def Wasserstein_loss_D(real, fake):
    """
    Wasserstein loss for the discriminator
    """
    loss = - th.mean(real) + th.mean(fake)
    return loss


def Wasserstein_loss_G(fake):
    """
    Wasserstein loss for the generator
    """
    loss = - th.mean(fake)
    return loss

def Wasserstein_loss_V(real, fake):
    """
    Wasserstein loss for validation
    """
    return Wasserstein_loss_D(fake, real)  # Just swap real and fake


def Wasserstein_gradient_penalty(real_data, fake_data, netD, alpha):
    """
    Wasserstein gradient penalty for the discriminator
    """
    # To add a gradient penality, first create an interpolate between real and fake data
    a_0 = th.rand(size=(real_data.shape[0], 1), device=real_data.device)
    a = a_0.repeat(1, real_data.shape[1])
    interpolated_data = a * real_data + (1 - a) * fake_data
    # Allow gradient computation over interpolated data
    interpolates = th.autograd.Variable(interpolated_data, requires_grad=True)
    # Feed interpolated data to the discriminator
    output_interpolates = netD(interpolates, alpha)
    # Compute gradients
    gradients = th.autograd.grad(outputs=output_interpolates, inputs=interpolates,
                                 grad_outputs=th.ones(output_interpolates.size()).to(interpolates.device),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
    # Compute penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def Wasserstein_drift_penalty(output):
    """
    Wasserstein drift penalty for the discriminator
    """
    return th.mean(output ** 2)


def L2_fft_loss(tensor, window, target, std, epsilon=1e-10):
    """
    L2 validation loss in frequency domain
    """
    tensor_fft = from_time_to_frequency(tensor, window)
    tensor_abs = from_fft_evaluate_abs(tensor_fft)
    target = target.repeat(tensor_abs.shape[0],1).to(tensor_fft.device)
    std[std < epsilon] = epsilon
    std = std.repeat(tensor_abs.shape[0], 1).to(tensor_fft.device)
    loss = th.mean(((tensor_abs - target) / std) ** 2)
    return loss
