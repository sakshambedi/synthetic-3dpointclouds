import numpy as np
import torch

# from torchvision.transforms import Compose


class RandomRotation:
    def __call__(self, pc):
        theta = np.random.uniform(0, 2 * np.pi)
        rot_matrix = torch.tensor(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ],
            dtype=pc.dtype,  # keep the same dtype as the input pc
        )

        # Separate xyz coordinates (first 3 columns) and other features
        xyz = pc[:, :3]
        rgb = pc[:, 3:]

        rotated_xyz = xyz @ rot_matrix  # Apply rotation only to xyz coordinates

        rotated_pc = torch.cat(
            (rotated_xyz, rgb), dim=1
        )  # contact rotated CP and RGB data

        return rotated_pc


class RandomJitter:
    """Add Gaussian noise to each point."""

    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, pc):
        jitter = torch.randn_like(pc) * self.sigma
        return pc + jitter


class ColorsToTensor:
    def __init__(self, normalize=True, divisor=255.0):
        """
        Args:
            normalize (bool): Whether to normalize the color values.
            divisor (float): The value by which to divide the colors if normalization is needed.
        """
        self.normalize = normalize
        self.divisor = divisor

    def __call__(self, pc):
        """
        Args:
            pc (np.ndarray or torch.Tensor): The input point cloud with shape (N, 6).
                The first 3 columns are coordinates and the last 3 are colors.
        Returns:
            torch.Tensor: The transformed point cloud with normalized colors.
        """

        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc)

        pc = pc.float()

        if self.normalize:
            colors = pc[:, 3:]
            # If colors are in [0, 255], their max value will be > 1.
            if colors.max() > 1.0:
                pc[:, 3:] = colors / self.divisor

        return pc


# transform = Compose([ColorsToTensor(), RandomRotation(), RandomJitter(sigma=0.02)])
