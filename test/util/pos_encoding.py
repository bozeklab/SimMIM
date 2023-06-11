import unittest
import torch
import numpy as np

from test_examples.mae_pos_encoding import get_2d_sincos_pos_embed
from util.pos_embedd import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):

    def test_encode_relative_position(self):
        grid_size = 2
        pos = torch.tensor([[1, 1, 2, 2, 1, 1, 2, 2],
                            [0, 0, 2, 2, 1, 1, 1, 1]], dtype=torch.float32)
        grid = torch.tensor([[[[0., 1.], [0., 1.]]], [[[0.,  0.], [1., 1.]]]], dtype=torch.float32)

        pos_enc = PositionalEncoding(4)
        encoded_grid = pos_enc.encode_relative_position(grid, pos, grid_size)
        encoded_grid = torch.tensor(encoded_grid)

        expected_encoded_grid = torch.tensor([[[[0.0, 1.0],
                                                [0.0, 1.0]],

                                               [[0.0, 0.0],
                                                [1.0, 1.0]]],

                                              [[[1.0, 1.5],
                                                [1.0, 1.5]],

                                               [[1.0, 1.0],
                                                [1.5, 1.5]]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(encoded_grid, expected_encoded_grid))

    def test_calculate_grid(self):
        grid_size = 2
        pos = torch.tensor([[1, 1, 2, 2, 1, 1, 2, 2],
                            [0, 0, 2, 2, 1, 1, 1, 1]], dtype=torch.float32)

        pos_enc = PositionalEncoding(4)
        encoded_grid1, encoded_grid2 = pos_enc.calculate_grid(pos, grid_size)

        expected_encoded_grid1 = torch.tensor([[[[0.0, 1.0],
                                                [0.0, 1.0]],

                                               [[0.0, 0.0],
                                                [1.0, 1.0]]],

                                              [[[0.0, 1.0],
                                                [0.0, 1.0]],

                                               [[0.0, 0.0],
                                                [1.0, 1.0]]]], dtype=torch.float32)

        expected_encoded_grid2 = torch.tensor([[[[0.0, 1.0],
                                                [0.0, 1.0]],

                                               [[0.0, 0.0],
                                                [1.0, 1.0]]],

                                              [[[1.0, 1.5],
                                                [1.0, 1.5]],

                                               [[1.0, 1.0],
                                                [1.5, 1.5]]]], dtype=torch.float32)

        self.assertTrue(torch.allclose(encoded_grid1, expected_encoded_grid1))
        self.assertTrue(torch.allclose(encoded_grid2, expected_encoded_grid2))

    def test_vs_mae(self):
        embed_dim = 128
        grid_size = 16

        pos_enc = PositionalEncoding(embed_dim=embed_dim)
        pos = torch.tensor([[0, 0, 16, 16, 5, 5, 11, 11],
                            [0, 0, 2, 2, 1, 1, 1, 1]], dtype=torch.float32)

        sequence_mae = get_2d_sincos_pos_embed(embed_dim, grid_size)
        sequence, _ = pos_enc(pos, grid_size)

        self.assertTrue(torch.allclose(torch.tensor(sequence_mae), sequence[0]))

    def test_get_2d_sincos_pos_embed_from_grid(self):
        embed_dim = 4
        grid = np.array([[[[0, 1], [2, 3]], [[4, 5], [6, 7]]]])

        batch_size, _, h, w = grid.shape

        pos_enc = PositionalEncoding(embed_dim)
        pos_embed = pos_enc.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

        expected_shape = (batch_size, h * w, embed_dim)
        self.assertEqual(pos_embed.shape, expected_shape)

    def test_get_1d_sincos_pos_embed_from_grid(self):
        embed_dim = 4
        pos = np.array([1, 2, 3, 4])

        pos_enc = PositionalEncoding(embed_dim)
        pos_embed = pos_enc.get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

        expected_shape = (4, embed_dim)
        self.assertEqual(pos_embed.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
