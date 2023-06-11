import numpy as np
import matplotlib.pyplot as plt
import torch

from test_examples.mae_pos_encoding import get_2d_sincos_pos_embed
from util.pos_embedd import PositionalEncoding


def main():
    # Example sequence of 16-dimensional vectors
    pos_enc = PositionalEncoding(embed_dim=128)
    pos = torch.tensor([[1, 1, 2, 2, 1, 1, 2, 2],
                        [0, 0, 2, 2, 1, 1, 1, 1]], dtype=torch.float32)

    #sequence, _ = pos_enc(pos, 16)

    sequence = get_2d_sincos_pos_embed(128, 16)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate the heatmap
    heatmap = ax.imshow(sequence.T, cmap='bwr', vmin=-1.0, vmax=1.0)

    # Set the colorbar
    _ = plt.colorbar(heatmap)

    # Set the axis labels
    ax.set_xlabel('sequence index')
    ax.set_ylabel('dimensions')

    # Set the title
    ax.set_title('sin-cos positional encoding')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()