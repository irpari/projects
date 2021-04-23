import torch as th
from WGAN.read_from_ini import read_train_parameters
from WGAN.trainer import Trainer
from WGAN.model import ProGANGenerator, ProGANDiscriminatorFourier
from WGAN.dataset import StrainDataset
from torch.utils.data import DataLoader, SubsetRandomSampler
import os


def train(train_folder, stage_max):

    # Import the device
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    # File name to save the parameter dict under
    file_name = train_folder + "train_param_dict.pt"
    if not os.path.exists(file_name):
        # It means we don't have to resume training from an old score dict
        param_dict = read_train_parameters(train_folder, save=True)
    else:
        param_dict = th.load(file_name, map_location=th.device("cpu"))

    # Initialize the dataset of real data
    path_to_dataset = param_dict["dataset"]["folder"] + param_dict["dataset"]["file_name"] + "_strain.pt"
    dataset = StrainDataset(path_to_dataset,
                            param_dict["dataset"]["max_length"],
                            param_dict["dataset"]["min_length"],
                            param_dict["dataset"]["overlap"],
                            param_dict["dataset"]["data_augmentation"])
    # Load the train indices
    indices = th.load(param_dict["dataset"]["folder"] + param_dict["dataset"]["file_name"] + "_indices.pt")
    indices_train = indices["train_indices"]
    param_dict["dataset"]["dataset_size"] = len(indices_train)
    # Initialize sampler
    train_sampler = SubsetRandomSampler(indices_train)
    # Create the dataloader
    train_dataloader = DataLoader(dataset,
                                  sampler=train_sampler,
                                  batch_size=param_dict["dataset"]["batch_size"],
                                  num_workers=param_dict["dataset"]["workers"],
                                  drop_last=True)

    # Set-up the validation dataset (make of fake signals, so sample from latent space)
    latent_size = param_dict["net_channels_G"][0]
    val_size = (int(len(dataset) * param_dict["dataset"]["validation_split"])) \
               // param_dict["dataset"]["batch_size"] * param_dict["dataset"]["batch_size"]
    if "validation_size" not in param_dict["dataset"].keys():
        # Add the validation size to param_dict
        param_dict["dataset"]["validation_size"] = val_size
    # Sample validation noise from latent space
    val_noise = th.normal(0, 1, size=(val_size, latent_size)).to(device)
    # Save it
    param_dict["dataset"]["validation_noise"] = val_noise

    # Import the fft data
    fft_dict = th.load(param_dict["dataset"]["folder"] + param_dict["dataset"]["file_name"] + "_fft_data.pt")

    # Initialize the nets
    net_G = ProGANGenerator(param_dict["net_channels_G"],  # List of the number of channels in each block
                            param_dict["kernel_conv"],  # Kernel size in convolutional layers
                            param_dict["dataset"]["min_length"],  # Length in time domain at stage 0
                            param_dict["generator_normalization"])  # Type of normalization to use

    net_D_f = ProGANDiscriminatorFourier(param_dict["net_channels_D"],  # List of the number of channels in each block
                                         param_dict["kernel_conv"],  # Kernel size in convolutional layers
                                         param_dict["dataset"]["min_length"],  # Length in time domain at stage 0
                                         param_dict["discriminator_blocks"])

    # Send the nets to device
    net_G.to(device)
    net_D_f.to(device)

    # Save the dictionary
    th.save(param_dict, file_name)

    # Initialize the trainer
    trainer = Trainer(net_D_f, net_G, train_dataloader, val_noise, param_dict,
                      fft_dict["train"], train_folder)

    trainer.train_to_stage(stage_max)