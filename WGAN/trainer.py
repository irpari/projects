import torch as th
from torch.optim import Adam
import copy
from WGAN.loss_functions import L2_fft_loss, Wasserstein_loss_D, Wasserstein_loss_G, \
    Wasserstein_gradient_penalty, Wasserstein_drift_penalty
from WGAN.plot_functions import plot_fake_signals, plot_losses, plot_convergence
from WGAN.read_from_ini import print_training_recap
from WGAN.fft_functions import hann_window
from numpy import prod
import os
import time


class Trainer:
    """
    Write Later
    """

    def __init__(self, net_D, net_G, train_dataloader, val_noise, param_dict, fft_dict, folder):
        """ Constructor for the class """
        super(Trainer, self).__init__()

        # Get the device
        self.device = val_noise.device

        # Check if we're resuming training from old scores
        self.path_to_scores = folder + "scores.pt"
        self.path_to_plots = folder + "plots/"
        if os.path.exists(self.path_to_scores):
            self.scores_dict = th.load(self.path_to_scores, map_location=self.device)
            resuming = True
        else:
            self.scores_dict = {}
            resuming = False
        if not os.path.exists(self.path_to_plots):
            os.mkdir(self.path_to_plots)

        # Do we want to print and/or write the results?
        self.printing = param_dict["printing"]
        self.writing = param_dict["writing"]

        # If we want to write down the logs, initialize the txt file
        if self.writing:
            self.path_to_txt = folder + "trainer_log.txt"
            if resuming:
                txt_file = open(self.path_to_txt, "a")
                text = "\n" + "-" * 120 + "\n" "Resuming training" + "\n" + "-" * 120 + "\n"
                txt_file.write(text)
                txt_file.close()
            else:  # Just create a new txt file
                txt_file = open(self.path_to_txt, "w+")
                txt_file.write("")
                txt_file.close()

        # If we're starting from stratch, print the recap of what we're doing
        if not resuming:
            print_training_recap(param_dict, self.printing, path_to_txt=self.path_to_txt if self.writing else None)

        # Save the nets
        self.net_D = net_D
        self.net_G = net_G

        # Save the dataloaders
        self.train_dataloader = train_dataloader

        # And the dataset fft information
        self.fft_dict = fft_dict
        # Save the validation dataset
        self.val_noise = val_noise
        # Size of validation dataset
        self.val_size = val_noise.shape[0]
        # Draw three random dataset element for plotting later
        self.rnd_idx = th.randint(0, len(train_dataloader.sampler), size=(9,))

        # Save the training parameters
        self.W_lambda_grad = param_dict["training"]["W_lambda_grad"]
        self.W_eps_drift = param_dict["training"]["W_eps_drift"]
        self.n_critic = param_dict["training"]["n_critic"]
        self.num_epochs_to_grow = param_dict["training"]["num_epochs_grow"]
        self.distance_to_achieve = param_dict["training"]["distance"]
        self.num_epochs_to_stop = param_dict["training"]["max_num_epochs"]
        self.patience = param_dict["training"]["patience"]
        self.plot_every_n = param_dict["training"]["plot_every_n"]
        self.save_every_n = param_dict["training"]["save_every_n"]

        # And the optimizer parameters
        self.opt_dict = param_dict["optimizer"]

        # Initialize the merging factor alpha and compute its increase rate
        if param_dict["training"]["num_epochs_grow"] != 0:
            self.d_alpha = 1 / param_dict["training"]["num_epochs_grow"]
        else:
            self.d_alpha = 1

        # Initialize or load the instance variables
        if not resuming:
            # Initialize the total number of iteration performed over the nets
            self.iter_G = 0
            self.iter_D = 0
            # Initialize the stage and status
            self.stage = 0
            self.status = "initialized"
            # Initialize the optimizers
            self.optimizer_D, self.optimizer_G = self.create_optimizers()
            # Initialize training time
            self.training_time = 0
        else:
            # Load the total number of iteration performed over the nets
            self.iter_G = self.scores_dict["iter_G"]
            self.iter_D = self.scores_dict["iter_D"]
            # Initialize the optimizers
            self.optimizer_D, self.optimizer_G = self.create_optimizers()
            # Load the stage and status
            self.stage = 0
            self.update_stage(self.scores_dict["stage"])
            self.status = self.scores_dict["status"]
            # And load old status of the optimizers and the nets
            self.load_state_dict_net_opt(self.scores_dict[self.stage], best=True)
            # Load training time
            self.training_time = self.scores_dict["training_time"]

        # Initialize windowing function for fft-s
        self.window = hann_window

    def update_stage(self, stage):
        stage_in = self.stage
        if stage - stage_in < 0:
            raise Exception("Cannot decrease training stage")
        # Add the blocks to the nets
        for i in range(stage - stage_in):
            # Add a new block to the nets and get a list containing the new parameters we've just added
            new_param_D = self.net_D.add_downsampling_block(self.device)
            new_param_G = self.net_G.add_upsampling_block(self.device)
            # Add the new parameters to the optimizers
            self.optimizer_D.param_groups[0]["params"].extend(new_param_D)
            self.optimizer_G.param_groups[0]["params"].extend(new_param_G)
        # Update the internal training stage
        self.train_dataloader.dataset.stage = stage
        self.stage = int(stage)
        return None

    def create_optimizers(self):
        optimizer_D = Adam(self.net_D.parameters(),
                           lr=self.opt_dict["lr_D"],
                           betas=self.opt_dict["betas"],
                           eps=self.opt_dict["eps"],
                           amsgrad=self.opt_dict["amsgrad"])
        optimizer_G = Adam(self.net_G.parameters(),
                           lr=self.opt_dict["lr_G"],
                           betas=self.opt_dict["betas"],
                           eps=self.opt_dict["eps"],
                           amsgrad=self.opt_dict["amsgrad"])
        return optimizer_D, optimizer_G

    def compute_L2_loss(self, item):
        dict_key = item.shape[-1]
        loss = L2_fft_loss(item.detach(),
                           self.window,
                           self.fft_dict[dict_key]["abs_mean"],
                           self.fft_dict[dict_key]["abs_std"])
        return loss.item()

    def load_state_dict_net_opt(self, old_scores, best=False):
        if not best:
            self.net_D.load_state_dict(old_scores["net_D"])
            self.net_G.load_state_dict(old_scores["net_G"])
            self.optimizer_D.load_state_dict(old_scores["opt_D"])
            self.optimizer_G.load_state_dict(old_scores["opt_G"])
        else:
            self.net_D.load_state_dict(old_scores["best_net_D"])
            self.net_G.load_state_dict(old_scores["best_net_G"])
            self.optimizer_D.load_state_dict(old_scores["best_opt_D"])
            self.optimizer_G.load_state_dict(old_scores["best_opt_G"])
        # Change lr accordingly to stage
        # for param_group in self.optimizer_D.param_groups:
        #     param_group['lr'] = self.opt_dict["lr_D"] / 2
        # for param_group in self.optimizer_G.param_groups:
        #     param_group['lr'] = self.opt_dict["lr_G"] / 2
        return None

    def train_to_stage(self, end):
        # Initialize
        keep_training = True
        start = self.stage  # Stage we're starting the training from
        # Check where we've left the training
        if self.status == "training":  # We haven't reached convergence at the current stage yet
            # So resume training from where we left it
            keep_training = self.train_one_stage(self.stage, end, self.scores_dict[self.stage])
        if self.status == "converged":
            start += 1

        if self.status == "converged" or self.status == "initialized":
            for stage in range(start, end + 1):
                if keep_training:
                    # In case one stage has failed convergence, keep_training breaks the loop
                    keep_training = self.train_one_stage(stage, end)
        return None

    def print_n_write(self, text, txt_file=None, flush=False):
        if self.printing:
            print(text)
        if txt_file is not None:
            txt_file.write(text + "\n")
            if flush:
                txt_file.close()
                txt_file = open(self.path_to_txt, "a+")
                return txt_file
        return None

    def save_weights(self, scores, best=True):
        if best:
            scores["best_net_D"] = copy.deepcopy(self.net_D.state_dict())  # Pass by value or die
            scores["best_net_G"] = copy.deepcopy(self.net_G.state_dict())
            scores["best_opt_D"] = copy.deepcopy(self.optimizer_D.state_dict())
            scores["best_opt_G"] = copy.deepcopy(self.optimizer_G.state_dict())
        else:
            scores["net_D"] = copy.deepcopy(self.net_D.state_dict())  # Pass by value or die
            scores["net_G"] = copy.deepcopy(self.net_G.state_dict())
            scores["opt_D"] = copy.deepcopy(self.optimizer_D.state_dict())
            scores["opt_G"] = copy.deepcopy(self.optimizer_G.state_dict())
        return None

    def check_signal_evolution(self, scores, stage):
        with th.no_grad():
            self.net_G.eval()
            fake_signal = self.net_G(self.val_noise[:10, :], scores["alpha"]).detach().cpu()
        small_fake = fake_signal[:9, :]
        small_real = th.cat([self.train_dataloader.dataset[self.rnd_idx[i].item()].unsqueeze(0)
                                    for i in range(0, 9)])
        scores["fake_signal_evolution"].append(fake_signal)
        scores["fake_signal_evolution_epochs"].append(scores["epoch"])
        plot_fake_signals(small_fake[:3, :].detach().cpu(), small_real[:3, :].detach().cpu(), self.window,
                          stage, scores["epoch"], self.fft_dict, self.path_to_plots)
        plot_convergence(small_fake[3:9, :].detach().cpu(), small_real[3:9, :].detach().cpu(), self.window,
                         stage, scores["epoch"], self.fft_dict,
                         printing=(self.path_to_plots, stage, scores["epoch"]))
        return None

    def plots_at_checkpoints(self, scores, stage):
        # Plot the losses as well
        plot_losses(scores["D_losses"], scores["G_losses"], scores["V_L2_losses"], self.distance_to_achieve,
                    self.num_epochs_to_grow, stage, self.path_to_plots)

        return None

    def train_one_stage(self, stage, end_stage=None, scores=None):
        if end_stage is None:
            end_stage = stage

        # Start time record
        training_time = - time.time()
        # Update all the internal stages
        self.update_stage(stage)

        # If we've to write down the progresses, open txt
        txt_file = open(self.path_to_txt, "a+") if self.writing else None

        # Compute the number of parameters in net D
        net_D_parameters = filter(lambda p: p.requires_grad, self.net_D.parameters())
        net_D_num_params = sum([prod(p.size()) for p in net_D_parameters])
        # Compute the number of parameters in net G
        net_G_parameters = filter(lambda p: p.requires_grad, self.net_G.parameters())
        net_G_num_params = sum([prod(p.size()) for p in net_G_parameters])

        # Print / write something
        separator = "-" * 60
        text = separator + "\n" + \
               "Training stage {} of {}".format(stage, end_stage) + \
               "\n" "Trainable parameters \t\t\tDiscriminator: {} \t\tGenerator: {}".format(net_D_num_params,
                                                                                            net_G_num_params)
        self.print_n_write(text, txt_file)

        # Check if we've to load old scores from checkpoint or reinitialize everything
        if scores is None:
            # Initialize dictionary
            scores = dict()
            # Evaluate the target for the validation loss
            # Initialize the lists to store the results
            scores["D_losses"] = []
            scores["G_losses"] = []
            scores["V_L2_losses"] = []
            scores["fake_signal_evolution"] = []
            scores["fake_signal_evolution_epochs"] = []
            # Initialize the counters
            scores["epoch"] = 0
            scores["patience"] = 0
            scores["best_distance"] = 1e8
            scores["last_plot"] = 0
            last_best_epoch = 0
            scores["alpha"] = None
            # Best weights / statuses
            self.save_weights(scores, best=True)
            # Last weights / statuses
            self.save_weights(scores, best=False)
            # Add status
            scores["status"] = "training"
            # Print / write something
            text = "New block starts with randomly initialized weights"

        else:
            # Load status of optimizers and nets as we left it at previous checkpoint
            self.load_state_dict_net_opt(scores, best=False)
            # Reset alpha
            self.net_D.alpha = scores["alpha"]
            self.net_G.alpha = scores["alpha"]
            # Update epoch
            last_best_epoch = scores["epoch"] - scores["patience"]
            scores["epoch"] += 1
            # Print / write something
            text = "Resuming from epoch {}".format(scores["epoch"])

        # Print / write something
        self.print_n_write(text, txt_file)

        # Initialize the convergence condition
        failed_convergence_condition = False
        # Reset the stop condition and the epoch
        stop_condition_reached = False

        self.print_n_write(separator, txt_file)
        while not stop_condition_reached and not failed_convergence_condition:
            # Update the merging parameter
            if scores["epoch"] <= self.num_epochs_to_grow and stage != 0:
                if scores["epoch"] == 0:
                    scores["alpha"] = self.d_alpha
                elif scores["epoch"] == self.num_epochs_to_grow:
                    scores["alpha"] = None
                    # Print / write something
                    text = separator + "\n" + \
                           "Growth of new block completed" + "\n" + \
                           separator
                    self.print_n_write(text, txt_file)
                else:
                    scores["alpha"] += self.d_alpha

            # Train the net and update the losses
            loss_D, loss_G, loss_V_L2 = self.train_one_epoch(scores["alpha"])

            # Append new loss values
            scores["D_losses"].append(loss_D)
            scores["G_losses"].append(loss_G)
            scores["V_L2_losses"].append(loss_V_L2)

            # Is it time to check convergence?
            if stage == 0 or scores["epoch"] >= self.num_epochs_to_grow:
                # Compute the distance from target
                distance = abs(loss_V_L2 - 1)
                if distance < scores["best_distance"]:  # Check if we're getting closer to the target
                    scores["best_distance"] = distance
                    # If so, update the best weights / statuses
                    self.save_weights(scores, best=True)
                    # And restore the patience
                    scores["patience"] = 0
                    last_best_epoch = scores["epoch"]
                    # Check if it's time to stop. To stop the following conditions must be satisfied:
                    #                                   - the distance from target must be smaller than required
                    #                                   - the growth of last block must be completed
                    stop_condition_reached = distance < self.distance_to_achieve

                else:  # We're losing patience here!
                    scores["patience"] += 1

                failed_convergence_condition = scores["epoch"] >= self.num_epochs_to_stop or \
                                               scores["patience"] >= self.patience

            else:  # If we're still training, update last best epoch anyways
                last_best_epoch = scores["epoch"]

            # Print / write this epoch results
            text = "[{:1d}/{:1d}] [Ep: {:4d}/{:d}]".format(stage, end_stage, scores["epoch"], self.num_epochs_to_stop)
            text += "[α {:.2f}]".format(scores["alpha"]) if scores["alpha"] is not None else \
                    "[p {:3d}]".format(scores["patience"])
            text += "\tLoss_D: {:8.3f} \t|  Loss_G: {:8.3f}".format(loss_D, loss_G) + \
                    "\t|  Loss_V_L2: {:10.3f}".format(loss_V_L2)

            self.print_n_write(text, txt_file)

            # Check if it is time to plot the signals
            if scores["epoch"] == 0 or (last_best_epoch - scores["last_plot"]) >= self.plot_every_n:
                # Plot
                self.check_signal_evolution(scores, stage)
                # Update last plot epoch
                scores["last_plot"] = scores["epoch"]

            # Check if it's time to stop the training or to save checkpoints
            if stop_condition_reached or failed_convergence_condition or \
                (scores["epoch"] % self.save_every_n == 0 and scores["epoch"] != 0):
                # Update last weights / statuses
                self.save_weights(scores, best=False)
                self.plots_at_checkpoints(scores, stage)

                # Update training time
                training_time += time.time()
                self.training_time += training_time
                days = int(self.training_time // (24 * 60 * 60))
                hours = int((self.training_time - 24 * 60 * 60 * days) // (60 * 60))
                min = int((self.training_time - 24 * 60 * 60 * days - 60 * 60 * hours) // 60)
                training_time = - time.time()

                # Save and print in which case we're (converged, not converged, just checkpoint)
                text = separator + "\n" + \
                       "Training time: {:d} days {:d} hours {:d} min \n".format(days, hours, min)

                if stop_condition_reached:  # The convergence criterion is satisfied
                    # Print / write something
                    self.status = "converged"
                    scores["status"] = "converged"
                    text += "End training stage {} of {}".format(stage, end_stage) + "\n" + \
                            "Convergence reached after epoch {} with distance {:.3f} from target."\
                               .format(scores["epoch"], distance) + \
                            "\n" + separator * 2 + \
                            "\n" + separator * 2

                elif failed_convergence_condition:  # We've exceeded the maximum number of trainable epochs
                    # Print / write something
                    self.status = "failed"
                    scores["status"] = "failed"
                    text += "Abort training at stage {} due to failed convergence \n".format(stage) + \
                            "Min distance from target {:.3f}, reached at epoch {}"\
                                .format(distance, scores["epoch"]) + \
                            "\n" + separator * 2 + \
                            "\n" + separator * 2

                else:
                    self.status = "training"
                    scores["status"] = "training"
                    # Print / write something
                    text += "Saving checkpoint at epoch {}".format(scores["epoch"]) + \
                            "\n" + separator

                txt_file = self.print_n_write(text, txt_file, flush=True)
                self.update_and_save_scores(scores)

            # Update the counter
            scores["epoch"] += 1

        # Close txt file
        if self.writing:
            txt_file.close()
        return False if failed_convergence_condition else True

    def update_and_save_scores(self, scores):
        # Add the losses of last stage as a sub-dictionary
        self.scores_dict[self.stage] = scores

        # If we've reached convergence, update the global losses as well
        self.scores_dict["status"] = self.status
        self.scores_dict["iter_D"] = self.iter_D
        self.scores_dict["iter_G"] = self.iter_G
        self.scores_dict["stage"] = self.stage
        self.scores_dict["training_time"] = self.training_time

        # Save
        th.save(self.scores_dict, self.path_to_scores)
        return None

    def train_one_epoch(self, alpha):
        # Train mode
        th.set_grad_enabled(True)
        self.net_D.train()
        self.net_G.train()

        # Reset the running loss
        loss_D = 0
        loss_G = 0
        # The number of iteration over G may vary from epochs to epochs
        iter_G_epoch = 0  # So we've to keep track of them too
        # Loop over real data
        for real_data in self.train_dataloader:
            # Update discriminator
            self.optimizer_D.zero_grad()
            # Get real data and send it to device
            real_data = real_data.to(self.device)
            # Feed it to the discriminator
            output_real = self.net_D(real_data, alpha)
            # Generate some noise
            noise = th.normal(0, 1, size=(real_data.shape[0],  # batch size
                                          self.val_noise.shape[1])).to(self.device)  # latent size
            # Use the generator and obtain some fake data
            fake_data = self.net_G(noise, alpha)
            # Feed it to the discriminator
            output_fake = self.net_D(fake_data.detach(), alpha)
            # (detached it because we don't need to back-propagate over G)e
            # Compute the Wasserstein loss for the discriminator
            batch_loss_D = Wasserstein_loss_D(output_real, output_fake)
            # Add a gradient penalty
            batch_loss_D += self.W_lambda_grad * Wasserstein_gradient_penalty(real_data, fake_data, self.net_D, alpha)
            # Add drift penalty
            batch_loss_D += self.W_eps_drift * Wasserstein_drift_penalty(output_real)
            # Backward propagation of the gradients
            batch_loss_D.backward()
            # Update the weights
            self.optimizer_D.step()
            # Store the loss
            loss_D += batch_loss_D.item()
            # Update the iterations over D
            self.iter_D += 1

            # Check if it's time to update the generator
            if self.iter_D % self.n_critic == 0:
                self.optimizer_G.zero_grad()
                # Feed the fake data we've generated last time to the updated discriminator
                output_fake = self.net_D(fake_data, alpha)  # Do not detach it this time!
                # Compute the Wasserstein loss for the generator
                batch_loss_G_Wass = Wasserstein_loss_G(output_fake)
                # Backward propagation of the gradients
                batch_loss_G_Wass.backward()
                # Update the weights
                self.optimizer_G.step()
                # Store the loss
                loss_G += batch_loss_G_Wass.item()
                # Update the iterations over G
                self.iter_G += 1
                iter_G_epoch += 1

        # Average running loss
        loss_D /= len(self.train_dataloader)
        loss_G /= iter_G_epoch
        # Perform validation
        with th.no_grad():
            self.net_D.eval()
            self.net_G.eval()
            fake_val = self.net_G(self.val_noise, alpha)
            loss_V_L2 = self.compute_L2_loss(fake_val)
        # Return the losses
        return loss_D, loss_G, loss_V_L2
