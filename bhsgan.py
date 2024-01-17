import torch.nn as nn

from utils import Positive, TanhScale


class GeneratorBhsSim(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


class DiscriminatorBhsSim(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
            nn.Linear(8, 1),
            nn.Softplus(),
        )

    def forward(self, input):
        return self.main(input)


class GeneratorBhsMnist(nn.Module):
    def __init__(self, z_dim=100, output_dim=28 * 28, hidden_dim=512):
        super().__init__()

        self.z_dim = z_dim

        self.main = nn.Sequential(
            self.get_generator_block(z_dim, hidden_dim),
            self.get_generator_block(
                hidden_dim,
                hidden_dim * 2,
            ),
            self.get_generator_final_block(
                hidden_dim * 2,
                output_dim,
            ),
        )

    def get_generator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def get_generator_final_block(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, x):
        return self.main(x)


class DiscriminatorBhsMnist(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=784):
        super().__init__()
        self.main = nn.Sequential(
            self.get_critic_block(
                input_dim,
                hidden_dim,
            ),
            self.get_critic_block(
                hidden_dim,
                hidden_dim // 2,
            ),
            self.get_critic_block(
                hidden_dim // 2,
                hidden_dim // 4,
            ),
            self.get_critic_final_block(
                hidden_dim // 4,
                1,
            ),
        )

    def get_critic_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.BatchNorm1d(output_dim),
            nn.ELU(inplace=True),
        )

    def get_critic_final_block(self, input_dim, output_dim):
        return nn.Sequential(nn.Linear(input_dim, output_dim), Positive())

    def forward(self, image):
        return self.main(image)


class GeneratorBhsLsun(nn.Module):
    def __init__(self, n_channels_out=3, image_size=64, z_dim=100):
        super().__init__()

        self.z_dim = z_dim

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            self.get_generator_block(
                input_channels=z_dim,
                out_channels=image_size * 16,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            # state size. ``(ngf*16) x 4 x 4``
            self.get_generator_block(
                input_channels=image_size * 16,
                out_channels=image_size * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # state size. ``(ngf*8) x 8 x 8``
            self.get_generator_block(
                input_channels=image_size * 8,
                out_channels=image_size * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # state size. ``(ngf*4) x 16 x 16``
            self.get_generator_block(
                input_channels=image_size * 4,
                out_channels=image_size * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # state size. ``(ngf*2) x 32 x 32``
            self.get_generator_final_block(
                input_channels=image_size * 2,
                out_channels=n_channels_out,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
            # state size. ``(nc) x 64 x 64``
        )

    def get_generator_block(
        self, input_channels, out_channels, kernel_size, stride, padding, bias
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def get_generator_final_block(
        self, input_channels, out_channels, kernel_size, stride, padding, bias
    ):
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.Sigmoid()(),
        )

    def forward(self, x):
        return self.main(x)


class DiscriminatorBhsLsun(nn.Module):
    def __init__(self, n_channels_in=3, image_size=64):
        super().__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            self.get_critic_block(
                input_channels=n_channels_in,
                out_channels=image_size,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # state size. ``(ndf) x 32 x 32``
            self.get_critic_block(
                input_channels=image_size,
                out_channels=image_size * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # state size. ``(ndf*2) x 16 x 16``
            self.get_critic_block(
                input_channels=image_size * 2,
                out_channels=image_size * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # state size. ``(ndf*4) x 8 x 8``
            self.get_critic_block(
                input_channels=image_size * 4,
                out_channels=image_size * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            # state size. ``(ndf*8) x 4 x 4``
            self.get_critic_final_block(
                input_channels=image_size * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

    def get_critic_block(
        self, input_channels, out_channels, kernel_size, stride, padding, bias
    ):
        return nn.Sequential(
            nn.Conv2d(
                input_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def get_critic_final_block(
        self, input_channels, out_channels, kernel_size, stride, padding, bias
    ):
        return nn.Sequential(
            nn.Conv2d(
                input_channels, out_channels, kernel_size, stride, padding, bias=bias
            ),
            Positive(),
        )

    def forward(self, image):
        return self.main(image)
