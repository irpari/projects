import torch as th
import matplotlib.pyplot as plt
from WGAN.fft_functions import from_time_to_frequency, from_fft_evaluate_abs, from_fft_evaluate_angle


def create_time_freq_array(stage, initial_length):
    fs = 4096
    # Create time and frequency array
    step = fs / (initial_length * 2 ** stage)
    time = th.arange(0, fs, step=step).float() / fs
    stride = (fs // initial_length) // (2 ** stage)
    f_sampling = fs / stride
    f_nq = f_sampling // 2
    step = f_nq / ((initial_length * 2 ** stage) // 2)
    freqs = th.arange(0, f_nq + step, step=step)
    return time, freqs


def plot_fake_signals(fake_signal, real_signal, window, stage, epoch, fft_dict, path):
    # Where to save the plots
    file_name = path + "s_{:d}_ep_{:04d}.png".format(stage, epoch)
    # Retrieve length at stage 0
    initial_length = fake_signal.shape[-1] / (2 ** stage)
    # Evaluate fft
    real_fft = from_time_to_frequency(real_signal, window)
    fake_fft = from_time_to_frequency(fake_signal, window)
    real_abs = from_fft_evaluate_abs(real_fft)
    fake_abs = from_fft_evaluate_abs(fake_fft)

    # Create time and frequency array
    time, freqs = create_time_freq_array(stage, initial_length)

    time_range_upper = 0.2 * th.ones(len(time))
    time_range_lower = - time_range_upper
    stage_key = len(time)
    abs_range_upper = fft_dict[stage_key]["abs_mean"] + fft_dict[stage_key]["abs_std"]
    abs_range_lower = abs_range_upper - 2 * fft_dict[stage_key]["abs_std"]
    plt.figure(figsize=(15, 10))

    plt_idx = 1
    for x in range(3):
        # Time domain plots
        plt.subplot(3, 3, plt_idx)
        plt.fill_between(time, time_range_lower, time_range_upper, color="lightskyblue", alpha=0.6)
        plt.axhline(0, color="lightskyblue", alpha=0.9, linestyle="--")
        plt.plot(time, real_signal[x], label="real")
        plt.plot(time, fake_signal[x], label="fake")
        plt.legend(loc=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Signal, rescaled")
        plt_idx += 1

    for x in range(3):
        # Frequency domain, abs
        plt.subplot(3, 3, plt_idx)
        plt.fill_between(freqs, abs_range_lower, abs_range_upper, color="lightskyblue", alpha=0.6)
        plt.plot(freqs, fft_dict[stage_key]["abs_mean"], color="lightskyblue", alpha=0.9,
                 linestyle="--")
        plt.loglog(freqs, real_abs[x], label="real")
        plt.loglog(freqs, fake_abs[x], label="fake")
        plt.legend(loc=2)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Signal, abs(DFT), rescaled")
        plt_idx += 1

    for x in range(3):
        real_residues = (real_abs[x] - fft_dict[stage_key]["abs_mean"]) / fft_dict[stage_key]["abs_std"]
        fake_residues = (fake_abs[x] - fft_dict[stage_key]["abs_mean"]) / fft_dict[stage_key]["abs_std"]
        # Frequency domain, angle
        plt.subplot(3, 3, plt_idx)
        plt.fill_between(freqs, - th.ones(len(freqs)), th.ones(len(freqs)), color="lightskyblue", alpha=0.6)
        plt.axhline(0, color="lightskyblue", alpha=0.9,linestyle="--")
        plt.plot(freqs, real_residues, label="real")
        plt.plot(freqs, fake_residues, label="fake")
        plt.legend(loc=2)
        plt.xscale("Log")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("residues(FFT)")
        plt_idx += 1

    # And title
    plt.suptitle("Stage {:d}, Epoch {:d}".format(stage, epoch), fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Save and don't forget to close
    plt.savefig(file_name)
    plt.close()
    return None


def plot_losses(D_losses, G_losses, V_L2_losses, distance, num_epochs_grow, stage, path):
    # Where to save the plots
    file_name = path + "s_{:d}_losses.png".format(stage)
    plt.figure(figsize=(15, 10))

    # Plot discriminators and generator
    plt.subplot(2, 1, 1)
    plt.axhline(0, alpha=0.15, c="k", linewidth=2)
    if stage != 0:
        plt.axvline(num_epochs_grow, alpha=0.7, c="k", linestyle="--", label="end growth", linewidth=2)
    plt.plot(D_losses, label="D loss", c="C00", linewidth=2)
    plt.plot(G_losses, label="G loss", c="C01", linewidth=2)
    plt.xlabel("time in epochs")
    plt.ylabel("loss")
    plt.legend(loc=1)
    plt.title("Discriminators and generator losses at stage {}".format(stage), fontsize=14, fontweight="bold")

    # Plot validation L2
    plt.subplot(2, 1, 2)
    target = 1
    upper_bound = target * th.ones(len(V_L2_losses)) + distance
    lower_bound = upper_bound - 2 * distance
    plt.fill_between(th.arange(len(V_L2_losses)), upper_bound, lower_bound, color="yellow")
    plt.plot(V_L2_losses, label="V L2 loss", c="C02")
    plt.axhline(target, alpha=0.4, c="k", linestyle="--", label="target", linewidth=3)
    plt.axhline(min(V_L2_losses), alpha=0.3, c="k", linestyle="dotted", label="best loss", linewidth=2)
    if stage != 0:
        plt.axvline(num_epochs_grow, alpha=0.8, c="k", linestyle="--", label="end growth", linewidth=2)
    plt.yscale("Log")
    plt.xlabel("time in epochs")
    plt.ylabel("loss")
    plt.legend(loc=1)
    plt.title("Validation L2 loss at stage {}".format(stage), fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save and don't forget to close
    plt.savefig(file_name)
    plt.close()
    return None


def plot_convergence(fake_signal, real_signal, window, stage, epoch, fft_dict, printing=None, title=None):
    if printing is None:
        plotting = True
    else:
        path, stage, epoch = printing
        file_name = path + "s_{:d}_convergence.png".format(stage, epoch)
        plotting = False

    initial_length = fake_signal.shape[-1] / (2 ** stage)  # Retrieve length at stage 0
    # Evaluate fft
    real_fft = from_time_to_frequency(real_signal, window)
    fake_fft = from_time_to_frequency(fake_signal, window)
    real_abs = from_fft_evaluate_abs(real_fft)
    fake_abs = from_fft_evaluate_abs(fake_fft)

    # Create  frequency array
    time, freqs = create_time_freq_array(stage, initial_length)
    stage_key = len(time)

    abs_range_upper = fft_dict[stage_key]["abs_mean"] + fft_dict[stage_key]["abs_std"]
    abs_range_lower = abs_range_upper - 2 * fft_dict[stage_key]["abs_std"]

    plt.figure(figsize=(20, 10))
    # Plot real data
    plt.subplot(2, 1, 1)
    for i in range(6):
        plt.loglog(freqs, real_abs[i])
    plt.fill_between(freqs, abs_range_lower, abs_range_upper, color="lightskyblue", alpha=0.3)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("abs(FFT)")
    plt.title("Real signals in frequency domain", fontsize=14)
    # Plot fake data
    plt.subplot(2, 1, 2)
    for i in range(6):
        plt.loglog(freqs, fake_abs[i])
    plt.fill_between(freqs, abs_range_lower, abs_range_upper, color="lightskyblue", alpha=0.3)
    plt.xlabel("frequency [Hz]")
    plt.ylabel("abs(FFT)")
    plt.title("Fake signals in frequency domain", fontsize=14)
    if title is None:
        title = "Convergence of stage {:d} after {:d} epochs".format(stage, epoch)
    plt.suptitle(title, fontsize=16,
                 fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if plotting:
        plt.show()
    else:
        plt.savefig(file_name)
        plt.close()
    return None
