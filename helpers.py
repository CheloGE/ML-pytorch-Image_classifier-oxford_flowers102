import torchvision
import torch
import numpy as np
class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = np.array(mean)
        std = np.array(std)
        std_inv = torch.Tensor(1 / (std + 1e-7))
        mean_inv = torch.Tensor(-mean / std)
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())