import torch
import matplotlib.pyplot as plt
import math

from lib.net.conv_net import ConvNavNet

def plot_conv1_filters(weights, title):
    # weights: (32, C, 8, 8)
    num_filters = weights.shape[0]
    C, H, W = weights.shape[1:]
    cols = 8
    rows = math.ceil(num_filters / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    axs = axs.flatten()

    for i in range(num_filters):
        w = weights[i]  # (C,8,8)

        # If RGB: show as color image
        if C == 3:
            # Normalize to [0,1] for visualization
            w_min, w_max = w.min(), w.max()
            img = (w - w_min) / (w_max - w_min + 1e-8)
            img = img.permute(1,2,0).numpy()  # (8,8,3)
            axs[i].imshow(img)
        else:
            axs[i].imshow(w[0].numpy(), cmap='gray')
        axs[i].axis('off')

    for j in range(i+1, len(axs)):
        axs[j].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example: compare early and late training
    w_ep100 = torch.load("conv1_filters_ep1000.pt")  # adjust filename
    w_ep1000 = torch.load("conv1_filters_ep10000.pt")

    plot_conv1_filters(w_ep100, "Conv1 filters after 1000 episodes")
    plot_conv1_filters(w_ep1000, "Conv1 filters after 5000 episodes")