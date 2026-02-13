import sys
import os
import torch
import unittest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MLP, CNN1D, BiLSTM

class TestModels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 100
        self.num_classes = 3
        self.input_tensor = torch.randn(self.batch_size, self.input_dim)

    def test_mlp_output_shape(self):
        model = MLP(self.input_dim, num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_cnn_output_shape(self):
        model = CNN1D(self.input_dim, num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

    def test_bilstm_output_shape(self):
        model = BiLSTM(self.input_dim, num_classes=self.num_classes)
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))

if __name__ == '__main__':
    unittest.main()
