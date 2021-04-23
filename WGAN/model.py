import torch as th
from WGAN.fft_functions import from_time_to_frequency, hann_window


# Custom layers, reference here: https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/CustomLayers.py
# https://en.wikipedia.org/wiki/Fan-in
class EqualizedLinear(th.nn.Module):
    """
    Linear (dense) layer modified for equalized learning rate
    by scaling the weights by the factor (2 / fan_in) ** 1 / 2
    """
    def __init__(self, c_in, c_out, bias=True):
        """ Constructor for the class """
        super(EqualizedLinear, self).__init__()

        # Compute the fan-in
        fan_in = c_in
        # And the scale factor
        self.scale = (2 / fan_in) ** 0.5

        # Initialize the weights: normal distribution
        self.weight = th.nn.Parameter(th.nn.init.normal_(th.empty(c_out, c_in)),
                                      requires_grad=True)
        # And the bias, if using it
        self.use_bias = bias
        if self.use_bias:
            self.bias = th.nn.Parameter(th.zeros(c_out).float(),  # Zero initialization
                                        requires_grad=True)

    def forward(self, x):
        return th.nn.functional.linear(x,
                                       self.weight * self.scale,  # Scaled weights here!
                                       self.bias if self.use_bias else None)


class EqualizedConv1d(th.nn.Module):
    """
    1D convolutional layer modified for equalized learning rate
    by scaling the weights by the factor (2 / fan_in) ** 1 / 2
    """
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        """ Constructor for the class """
        super(EqualizedConv1d, self).__init__()
        # Define stride and padding
        self.stride = stride
        self.padding = padding
        # Compute the fan-in
        fan_in = kernel_size * c_in  # value of fan_in
        # And the scale factor
        self.scale = (2 / fan_in) ** 0.5
        # Initialize the weights: normal distribution
        self.weight = th.nn.Parameter(th.nn.init.normal_(th.empty(c_out, c_in, kernel_size)),
                                      requires_grad=True)
        # And the bias, if using it
        self.use_bias = bias
        if self.use_bias:
            self.bias = th.nn.Parameter(th.zeros(c_out).float(),  # Zero initialization
                                        requires_grad=True)

    def forward(self, x):
        return th.nn.functional.conv1d(input=x,
                                       weight=self.weight * self.scale,  # Scaled weights here!
                                       bias=self.bias if self.use_bias else None,
                                       stride=self.stride,
                                       padding=self.padding)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


# https://towardsdatascience.com/what-is-transposed-convolutional-layer-40e5e6e31c11
class EqualizedDeconv1d(th.nn.Module):
    """
    1D transpose convolutional layer modified for equalized learning rate
    by scaling the weights by the factor (2 / fan_in) ** 1 / 2
    """
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, bias=True):
        """ Constructor for the class """
        super(EqualizedDeconv1d, self).__init__()
        # Define stride and padding
        self.stride = stride
        self.padding = padding
        # Compute the fan-in
        fan_in = c_in * kernel_size
        # And the scale factor
        self.scale = (2 / fan_in) ** 0.5
        # Initialize the weights: normal distribution
        self.weight = th.nn.Parameter(th.nn.init.normal_(th.empty(c_in, c_out, kernel_size)),
                                      requires_grad=True)
        # And the bias, if using it
        self.use_bias = bias
        if self.use_bias:
            self.bias = th.nn.Parameter(th.zeros(c_out).float(),  # Zero initialization
                                        requires_grad=True)

    def forward(self, x):
        return th.nn.functional.conv_transpose1d(input=x,
                                                 weight=self.weight * self.scale,  # Scaled weights here!
                                                 bias=self.bias if self.use_bias else None,
                                                 stride=self.stride,
                                                 padding=self.padding)

    def extra_repr(self):
        return ", ".join(map(str, self.weight.shape))


class PixelwiseNorm(th.nn.Module):
    """
    Pixelwise Normalization, alternative to batch normalization without learned parameters
    """
    def __init__(self, epsilon=1e-8):
        """ Constructor for the class """
        super(PixelwiseNorm, self).__init__()
        self.epsilon = epsilon  # avoid exploding denominators

    def forward(self, x):
        # Compute the root mean squared value over the channels
        # from [B, C, N] compute rms over C and get [B, 1, N]
        scale = x.pow(2.).mean(dim=1, keepdim=True).add(self.epsilon).sqrt()
        # Normalize and return
        y = x / scale
        return y


class MinibatchStdDev(th.nn.Module):
    """
    Minibatch standard deviation layer for the discriminator.
    Add a new channel constant over all the pixels consisting in the std evaluated over the batch and averaged over
    the channels and the pixels. Allows the discriminator to discern between a fake and a real batch as the former
    tends to have a smaller std (the generator lacks in creativity!).
    """

    def __init__(self, epsilon=1e-8):
        """ Constructor for the class """
        super(MinibatchStdDev, self).__init__()
        self.epsilon = epsilon  # avoid exploding denominators

    def forward(self, x):
        batch_size, channels, length = x.shape

        # [B x C x L] Subtract mean over batch
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x L]  Compute standard deviations over batch
        y = th.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + self.epsilon)
        # Do NOT use th.std, it gives you random nan-s and I've no idea where they come from

        # [1] Average over channels and pixels
        y = y.mean().view(1, 1, 1)

        # [B x 1 x N]  Replicate over batch and pixels
        y = y.repeat(batch_size, 1, length)

        # [B x (C + 1) x N]  Append as new channel
        y = th.cat([x, y], 1)
        return y


class UpsamplingBlock(th.nn.Module):
    """
    Up-scaling block for ProGAN Generator. Increases pixel dimension by a factor of 2 while taking the number of
    channels from channels_in to channels_out. Activation: LeakyReLU. Normalization: Pixelwise norm.
    If the time domain output is needed set last to True.
    If merging with the previous block is needed, pass the alpha parameter in the forward.
    """
    def __init__(self, channels_in, channels_out, conv_k, normalization, zero=False):
        """ Constructor for the class """
        super(UpsamplingBlock, self).__init__()
        # Initialize the layers
        self.zero = zero
        self.last = True
        self.second_to_last = False
        self.l1_conv = EqualizedConv1d(channels_in,
                                       channels_out,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1)//2,
                                       bias=True)
        self.l2_conv = EqualizedConv1d(channels_out,
                                       channels_out,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1)//2,
                                       bias=True)
        self.l3_conv = EqualizedConv1d(channels_out,
                                       channels_out,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1)//2,
                                       bias=True)
        self.l4_out = EqualizedConv1d(channels_out,
                                      1,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.act = th.nn.LeakyReLU(0.2, inplace=True)
        if normalization == "BatchNorm":
            self.l1_norm = th.nn.BatchNorm1d(channels_out)
            self.l2_norm = th.nn.BatchNorm1d(channels_out)
            self.l3_norm = th.nn.BatchNorm1d(channels_out)
        if normalization == "PixelWise":
            self.l1_norm = PixelwiseNorm()
            self.l2_norm = PixelwiseNorm()
            self.l3_norm = PixelwiseNorm()

    def forward(self, x, alpha=None):
        # Let's call:               B       batch size
        #                           C_in    channels in (n-1)-th block
        #                           C_out   channels in the n-th block (this one)
        #                           N0      starting size at the 0-th block
        # For the zeroth block:     C_out = L latent size

        # Check if we've to unpack
        if self.last and alpha is not None:
            x, y_out_old = x
        if not self.zero:  # x starts with shape [B, C_in, N0 * 2 ** (n-1)]
            # Upscale x to shape [B, C_in, # N0 * 2 ** n]
            y = th.nn.functional.interpolate(x,
                                             scale_factor=2,
                                             mode="linear",
                                             align_corners=True)
        else:
            y = x

        # First convolution + activation + normalization
        y = self.l1_norm(self.act(self.l1_conv(y)))  # [B, C_out, N0 * 2 ** n]
        # Second convolution + activation + normalization
        y = self.l2_norm(self.act(self.l2_conv(y)))  # [B, C_out, N0 * 2 ** n]
        # Third convolution + activation + normalization
        y = self.l3_norm(self.act(self.l3_conv(y)))  # [B, C_out, N0 * 2 ** n]
        if self.last:
            # Reduce channels to 1
            y_out = self.l4_out(y)  # [B, 1, N0 * 2 ** n]
            if alpha is not None:
                return alpha * y_out.squeeze(1) + (1 - alpha) * y_out_old.squeeze(1)
            else:
                return y_out.squeeze(1)
        elif self.second_to_last and alpha is not None:
            y_out = self.l4_out(y)  # [B, 1, N0 * 2 ** n]
            y_out = th.nn.functional.interpolate(y_out,
                                                 scale_factor=2,
                                                 mode="linear",
                                                 align_corners=True)
            return y, y_out
        return y


class ProGANGenerator(th.nn.Module):
    """
    A progressive GAN generator from https://arxiv.org/pdf/1710.10196.pdf adapted to time domain (1D) signals which
    can be growth to an arbitrary number of blocks.
    """
    def __init__(self, channels, kernel_sizes, initial_size, normalization):
        """ Constructor for the class """
        super(ProGANGenerator, self).__init__()
        # NOTE: counting the blocks from the top down. Meaning that the 0-th block is the one which receives the
        # input from the latent space and the 8-th layer is the one which, once the net is fully grown, outputs to
        # the time domain.

        # Instance the channels and the kernel sizes
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        # Instance the normalization
        self.normalization = normalization
        # Add the input layer (the one which gets the input from latent size)
        self.hypersphere_sampling = PixelwiseNorm()
        self.l0_conv = EqualizedDeconv1d(channels[0],  # Latent size corresponds to the channels in the 0th block
                                         channels[0],
                                         kernel_size=initial_size,  # channels_in gives the initial time length
                                         stride=1,
                                         padding=0,
                                         bias=True)
        self.act = th.nn.LeakyReLU(0.2, inplace=True)
        # Initialize the block list
        self.block_list = []
        self.Block = UpsamplingBlock
        # Add the zeroth block
        self.block_list.append(self.Block(channels[0],
                                          channels[0],
                                          kernel_sizes[0],
                                          self.normalization,
                                          zero=True))
        self.block_sequential = SequentialWithMultipleInputs(*self.block_list)
        # Instance the normalization
        if normalization == "BatchNorm":
            self.norm = th.nn.BatchNorm1d(channels[0])
        if normalization == "PixelWise":
            self.norm = PixelwiseNorm()

    def add_upsampling_block(self, device):
        # Get the new block index
        module_idx = len(self.block_list)
        # Initialize the new block
        new_block = self.Block(self.channels[module_idx-1],
                               self.channels[module_idx],
                               self.kernel_sizes[module_idx],
                               self.normalization).to(device)
        # Add it to the end of the block list
        self.block_list.append(new_block)
        # Check the new ordering of blocks
        for idx in range(len(self.block_list)):
            self.block_list[idx].last = False if idx != module_idx else True
            self.block_list[idx].second_to_last = False if idx != (module_idx - 1) else True
        # Update the sequence
        self.block_sequential = SequentialWithMultipleInputs(*self.block_list)
        # Also gets the new parameters to feed to the optimizer
        param_list = [param for param in new_block.parameters()]
        return param_list

    def forward(self, x, alpha):
        x = self.hypersphere_sampling(x)  # Convert white noise into points on a hypersphere
        x = x.unsqueeze(-1)
        y = self.norm(self.act(self.l0_conv(x)))  # Convert noise to channels initialization
        y_out = self.block_sequential(y, alpha)
        return y_out


class ProGANDiscriminatorFourier(th.nn.Module):
    """
    A progressive GAN discriminator from https://arxiv.org/pdf/1710.10196.pdf adapted to both time and Fourier domain
    for (1D) signals, which can be growth to an arbitrary number of blocks.
    """
    def __init__(self, channels, kernel_sizes, initial_size, block_type):
        """ Constructor for the class """
        super(ProGANDiscriminatorFourier, self).__init__()
        # Instance the branch in frequency domain
        self.frequency_branch = ProGANDiscriminatorFrequencyBranch(channels[::-1],
                                                                   kernel_sizes[::-1],
                                                                   initial_size,
                                                                   block_type)
        # Instance the branch in time domain
        self.time_branch = ProGANDiscriminatorTimeBranch(channels,
                                                         kernel_sizes,
                                                         initial_size,
                                                         block_type)
        # Output layer
        self.output_dense = EqualizedLinear(initial_size * 2, 1)
        # Create worker lists
        # self.worker_list = []
        # self.worker_list.append(Worker(self.frequency_branch))  # Working on the frequency branch
        # self.worker_list.append(Worker(self.time_branch))  # Working on the time branch
        # # Start workers
        # for worker in self.worker_list:
        #     worker.start()

    def add_downsampling_block(self, device):
        param_list = self.frequency_branch.add_downsampling_block(device)
        param_list += self.time_branch.add_downsampling_block(device)
        return param_list

    def forward(self, x, alpha):
        # Send process request
        # for worker in self.worker_list:
        #     worker.send_process_request((x, alpha))

        # Initialize the outputs
        out_tensors = [self.frequency_branch(x, alpha),
                       self.time_branch(x, alpha)]
        # completed_processes = 0
        # # Get outputs
        # while completed_processes != len(self.worker_list):
        #     for i, worker in enumerate(self.worker_list):
        #         out_tensor = worker.get_output_data()
        #         if out_tensor is not None:
        #             out_tensors[i] = out_tensor
        #             completed_processes += 1

        w = th.cat(out_tensors, dim=1)
        # And then through the output dense
        w = self.output_dense(w)
        return w


class ProGANDiscriminatorTimeBranch(th.nn.Module):
    """
    A progressive GAN discriminator from https://arxiv.org/pdf/1710.10196.pdf adapted to both time and Fourier domain
    for (1D) signals, which can be growth to an arbitrary number of blocks.
    """
    def __init__(self, channels, kernel_sizes, initial_size, block_type):
        """ Constructor for the class """
        super(ProGANDiscriminatorTimeBranch, self).__init__()
        # NOTE: counting the blocks from the bottom up. Meaning that the 0-th block is the one which outputs the signal
        # classification and the 8-th layer is the one which, once the net is fully grown, receives the nput from the
        # time domain.

        # Instance the channels and the kernels
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        # Instance the block type
        self.block_type = block_type

        # Create the lists of blocks
        self.block_list = []
        self.Block = DownsamplingBlockTimeDomain
        # Add zeroth block
        self.block_list.append(self.Block(channels[0],
                                          channels[0],
                                          kernel_sizes[0],
                                          self.block_type,
                                          zero=True))
        self.block_sequential = SequentialWithMultipleInputs(*self.block_list)

        # Initialize the output layer(s)
        self.output_conv = EqualizedConv1d(channels[0],
                                           channels[0],
                                           kernel_size=initial_size,
                                           stride=1,
                                           padding=0)
        self.output_dense = EqualizedLinear(channels[0],
                                            initial_size)
        # Always the same activation
        self.act = th.nn.LeakyReLU(0.2, inplace=True)

    def add_downsampling_block(self, device):
        # Get the new block index
        module_idx = len(self.block_list)
        # Initialize the new block
        new_block = self.Block(self.channels[module_idx],
                               self.channels[module_idx - 1],
                               self.kernel_sizes[module_idx],
                               self.block_type).to(device)
        # Add it at the beginning of the block list
        self.block_list.insert(0, new_block)
        # Also gets the new parameters to feed to the optimizer
        param_list = [param for param in new_block.parameters()]
        # Check the new ordering of blocks
        for idx in range(len(self.block_list)):
            self.block_list[idx].first = False if idx != 0 else True
            self.block_list[idx].second = False if idx != 1 else True
        # Update sequential
        self.block_sequential = SequentialWithMultipleInputs(*self.block_list)

        return param_list

    def forward(self, x, alpha):
        x = x.unsqueeze(1)
        x = self.block_sequential(x, alpha)
        x = self.act(self.output_conv(x))
        x = x.squeeze(-1)
        x = self.output_dense(x)
        return x


class SequentialWithMultipleInputs(th.nn.Sequential):
    def forward(self, x, alpha):
        for module in self._modules.values():
            x = module(x, alpha)
        return x


class ProGANDiscriminatorFrequencyBranch(th.nn.Module):
    """
    A progressive GAN discriminator from https://arxiv.org/pdf/1710.10196.pdf adapted to both time and Fourier domain
    for (1D) signals, which can be growth to an arbitrary number of blocks.
    """
    def __init__(self, channels, kernel_sizes, initial_size, block_type):
        """ Constructor for the class """
        super(ProGANDiscriminatorFrequencyBranch, self).__init__()
        # NOTE: counting the blocks from the bottom up. Meaning that the 0-th block is the one which outputs the signal
        # classification and the 8-th layer is the one which, once the net is fully grown, which receives the nput from
        # the time domain.

        # Instance the channels, the size at stage 0 and the kernel sizes
        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.initial_size = initial_size
        # Instance the block type
        self.block_type = block_type
        # Initialize window
        self.window = hann_window

        # Create the lists of blocks
        self.block_list = []
        self.Block = DownsamplingBlockFrequencyDomain
        # Add zeroth block
        self.block_list.append(self.Block(channels[0],
                                          channels[0],
                                          kernel_sizes[0],
                                          self.initial_size,
                                          self.block_type,
                                          zero=True))
        self.block_sequential = SequentialWithMultipleInputs(*self.block_list)

        # Add input layer for frequency domain
        self.input_conv_freq = EqualizedConv1d(2,
                                               channels[0],
                                               kernel_size=1,
                                               stride=1,
                                               padding=0)

        # Always the same activation
        self.act = th.nn.LeakyReLU(0.2, inplace=True)

    def add_downsampling_block(self, device):
        # Get the new block index
        module_idx = len(self.block_list)
        # Initialize the new block
        new_block = self.Block(self.channels[module_idx - 1],
                               self.channels[module_idx],
                               self.kernel_sizes[module_idx],
                               self.initial_size,
                               self.block_type).to(device)
        # Add it to the end of the block list
        self.block_list.append(new_block)
        # Also gets the new parameters to feed to the optimizer
        param_list = [param for param in new_block.parameters()]
        # Check the new ordering of blocks
        for idx in range(len(self.block_list)):
            self.block_list[idx].last = False if idx != module_idx else True
            self.block_list[idx].second_to_last = False if idx != (module_idx - 1) else True
        # Update sequential
        self.block_sequential = SequentialWithMultipleInputs(*self.block_list)
        return param_list

    def forward(self, x, alpha):
        y = from_time_to_frequency(x, self.window)  # Get fourier transform
        y = self.act(self.input_conv_freq(y))  # Go through the input layer
        z = self.block_sequential(y, alpha)
        return z


class DownsamplingBlockFrequencyDomain(th.nn.Module):
    """
    Down-sampling block for ProGAN Discriminator with optional residual connections. Choose between: ReZero,
    ResNet, ProGAN (no residual connections). Decrease pixel dimension by a factor of 2 while taking the number of
    channels from channels_in to channels_out. Activation: LeakyReLU. Normalization: None.
    If the input is directly from the frequency domain, set first to True in the forward.
    If still merging last two blocks, pass to them the alpha parameter in the forward.
    """
    def __init__(self, channels_in, channels_out, conv_k, initial_size, block_type, zero=False):
        """ Constructor for the class """
        super(DownsamplingBlockFrequencyDomain, self).__init__()
        # Save attributes
        self.zero = zero
        self.block_type = block_type

        # Same activation for all layers
        self.act = th.nn.LeakyReLU(0.2, inplace=True)

        # Initialize the layers

        self.l1_conv = EqualizedConv1d(channels_in,
                                       channels_in,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1)//2)

        self.l2_conv = EqualizedConv1d(channels_in,
                                       channels_out,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1)//2)

        self.l3_conv = EqualizedConv1d(channels_out,
                                       channels_out,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1)//2)

        self.l4_out_conv = EqualizedConv1d(channels_out,
                                           channels_out,
                                           kernel_size=initial_size // 2 + 1,
                                           stride=1,
                                           padding=0)
        self.l5_out_dense = EqualizedLinear(channels_out,
                                            initial_size)

        self.last = True
        self.second_to_last = False

        if self.block_type == "ReZero":  # Add extra learnable parameters and identity maps
            self.l1_beta = th.nn.Parameter(th.tensor([0]).float(),
                                           requires_grad=True)
            self.l2_beta = th.nn.Parameter(th.tensor([0]).float(),
                                           requires_grad=True)
            self.l3_beta = th.nn.Parameter(th.tensor([0]).float(),
                                           requires_grad=True)
            if channels_in != channels_out:
                self.l2_identity = EqualizedConv1d(channels_in,
                                                   channels_out,
                                                   kernel_size=1,
                                                   stride=1,
                                                   padding=0)
            else:
                self.l2_identity = th.nn.Identity()

        if self.block_type == "ResNet":
            if channels_in != channels_out:
                self.identity = EqualizedConv1d(channels_in,
                                                   channels_out,
                                                   kernel_size=1,
                                                   stride=1,
                                                   padding=0)
            else:
                self.identity = th.nn.Identity()

        # Padding is needed because the number of frequencies is odd
        self.downsample = th.nn.AvgPool1d(kernel_size=2, padding=1)

    def forward(self, x, alpha):
        # Let's call:       B       batch size
        #                   C_in    channels in n-th block (this one)
        #                   C_out   channels in the (n-1)-th block
        #                   N0      starting size at the 0-th block
        #                   s       stage of training
        #                   n       block index
        if self.last and alpha is not None:
            x, x_out = x  # Unpack

        if self.zero:
            y0 = x
        else:
            # Otherwise downsample it
            y0 = self.downsample(x)  # [B, C_in, N0 * 2 ** (s - n)]

        if self.block_type == "ReZero":
            # First convolution + activation
            y1 = y0 + self.l1_beta * self.act(self.l1_conv(y0))  # [B, C_in, N0 * 2 ** (s - n)]
            # Second convolution + activation
            y2 = self.l2_identity(y1) + self.l2_beta * self.act(self.l2_conv(y1))  # [B, C_out, N0 * 2 ** (s - n)]
            # Third convolution + activation
            y3 = y2 + self.l3_beta * self.act(self.l3_conv(y2))

        elif self.block_type == "ProGAN":
            # First convolution + activation
            y1 = self.act(self.l1_conv(y0))  # [B, C_in, N0 * 2 ** (s - n)]
            # Second convolution + activation
            y2 = self.act(self.l2_conv(y1))  # [B, C_out, N0 * 2 ** (s - n)]
            # Third convolution + activation
            y3 = self.act(self.l3_conv(y2))

        elif self.block_type == "ResNet":
            # First convolution + activation
            y1 = self.act(self.l1_conv(y0))  # [B, C_in, N0 * 2 ** (s - n)]
            # Second convolution + activation
            y2 = self.act(self.l2_conv(y1))  # [B, C_out, N0 * 2 ** (s - n)]
            # Third convolution + activatio
            y3 = self.act(self.identity(y0) + self.l3_conv(y2))

        if self.last or (self.second_to_last and alpha is not None):
            if self.last and alpha is None:
                y4 = self.act(self.l4_out_conv(y3)).squeeze(-1)
                return self.l5_out_dense(y4)
            elif self.last and alpha is not None:
                y4 = self.act(self.l4_out_conv(y3)).squeeze(-1)
                y_out = self.l5_out_dense(y4)
                return alpha * y_out + (1 - alpha) * x_out
            else:
                y4 = self.act(self.l4_out_conv(self.downsample(y3))).squeeze(-1)
                y_out = self.l5_out_dense(y4)
                return y3, y_out
        return y3


class DownsamplingBlockTimeDomain(th.nn.Module):
    """
    Down-sampling block for ProGAN Discriminator with ReZero residual connections. Decrease pixel dimension by a factor
    of 2 while taking the number of channels from channels_in to channels_out. Activation: LeakyReLU. Normalization:
    Pixelwise norm.
    If the input is from the time domain, set first to True in the forward.
    If merging with the previous block is needed, pass the alpha parameter in the forward.
    """
    def __init__(self, feature_in, feature_out, conv_k, block_type, zero=False):
        """ Constructor for the class """
        super(DownsamplingBlockTimeDomain, self).__init__()
        self.zero = zero
        self.first = True
        self.second = False
        self.block_type = block_type
        # Initialize the layers
        self.l0_input_conv = EqualizedConv1d(1,
                                             feature_in,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)
        self.act = th.nn.LeakyReLU(0.2, inplace=True)
        self.l1_conv = EqualizedConv1d(feature_in,
                                       feature_in,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1)//2)

        self.l2_conv = EqualizedConv1d(feature_in,
                                       feature_out,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1) // 2)

        self.l3_conv = EqualizedConv1d(feature_out,
                                       feature_out,
                                       kernel_size=conv_k,
                                       stride=1,
                                       padding=(conv_k - 1) // 2)

        if self.block_type == "ReZero":  # Add extra learnable parameters and identity maps
            self.l1_beta = th.nn.Parameter(th.tensor([0]).float(),
                                           requires_grad=True)
            self.l2_beta = th.nn.Parameter(th.tensor([0]).float(),
                                           requires_grad=True)
            self.l3_beta = th.nn.Parameter(th.tensor([0]).float(),
                                           requires_grad=True)
            if feature_in != feature_out:
                self.identity = EqualizedConv1d(feature_in,
                                                feature_out,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)
            else:
                self.identity = th.nn.Identity()

        if self.block_type == "ResNet":
            if feature_in != feature_out:
                self.identity = EqualizedConv1d(feature_in,
                                                feature_out,
                                                kernel_size=1,
                                                stride=1,
                                                padding=0)
            else:
                self.identity = th.nn.Identity()

        # No padding needed because the time samples are even
        self.downsample = th.nn.AvgPool1d(kernel_size=2, padding=0)

    def forward(self, x, alpha=None):
        # Let's call:       B       batch size
        #                   C_in    channels in n-th block (this one)
        #                   C_out   channels in the (n-1)-th block
        #                   N0      starting size at the 0-th block

        if self.second and alpha is not None:
            x, x_old = x
            x_old = self.downsample(x_old)
            x_new = self.downsample(x)
            y0 = (1 - alpha) * self.act(self.l0_input_conv(x_old)) + alpha * x_new
        elif self.first:
            y0 = self.act(self.l0_input_conv(x))
        else:
            y0 = self.downsample(x)  # [B, C_in, N0 * 2 ** n]

        if self.block_type == "ReZero":
            # First convolution + activation
            y1 = y0 + self.l1_beta * self.act(self.l1_conv(y0))  # [B, C_in, N0 * 2 ** n]
            y2 = self.identity(y1) + self.l2_beta * self.act(self.l2_conv(y1))  # [B, C_out, N0 * 2 ** n]
            y3 = y2 + self.l3_beta * self.act(self.l3_conv(y2))

        elif self.block_type == "ProGAN":
            # First convolution + activation
            y1 = self.act(self.l1_conv(y0))   # [B, C_in, N0 * 2 ** n]
            y2 = self.act(self.l2_conv(y1))   # [B, C_out, N0 * 2 ** n]
            y3 = self.act(self.l3_conv(y2))

        elif self.block_type == "ResNet":
            y1 = self.act(self.l1_conv(y0))   # [B, C_in, N0 * 2 ** n]
            y2 = self.act(self.l2_conv(y1))   # [B, C_out, N0 * 2 ** n]
            y3 = self.act(self.identity(y0) + self.l3_conv(y2))

        if self.first and alpha is not None:
            return y3, x
        else:
            return y3
