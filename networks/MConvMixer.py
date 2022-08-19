import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Down_Sample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Down_Sample, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), groups=in_dim, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=(1, 1), bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.conv(self.pool(x))


class Conv_Mixer_Layer(nn.Module):
    def __init__(self, dim, depth, kernel_size):
        super(Conv_Mixer_Layer, self).__init__()
        self.conv_mixer = nn.Sequential(
            *[nn.Sequential(Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding='same'),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)])

    def forward(self, x):
        x = self.conv_mixer(x)
        return x


class MConvMixer(nn.Module):
    def __init__(self, dims=(384, 768, 1536), depths=(6, 6, 8),
                 patch_size=(7, 7), n_class=6):
        super(MConvMixer, self).__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dims[0])
        )

        self.conv_mixer1 = Conv_Mixer_Layer(dim=dims[0], depth=depths[0], kernel_size=(9, 9))
        self.conv_mixer2 = Conv_Mixer_Layer(dim=dims[1], depth=depths[1], kernel_size=(9, 9))
        self.conv_mixer3 = Conv_Mixer_Layer(dim=dims[2], depth=depths[2], kernel_size=(7, 7))

        self.pool1 = Down_Sample(in_dim=dims[0], out_dim=dims[1])
        self.pool2 = Down_Sample(in_dim=dims[1], out_dim=dims[2])

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dims[2], n_class)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.conv_mixer1(x)
        x = self.pool1(x)
        x = self.conv_mixer2(x)
        x = self.pool2(x)
        x = self.conv_mixer3(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import torch

    t = torch.zeros(4, 3, 224, 224)
    m = MConvMixer()
    o = m(t)
    print(o.shape)
    total_params = sum(p.numel() for p in m.parameters())
    print("total para: {:.2f}".format(total_params/1e6))
