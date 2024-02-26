import torch
import torch.nn as nn

# from archs.arch_util import make_layer
# from archs.tenet.common import ConvBlock
# from utils.registry import ARCH_REGISTRY
# from archs.edvr_arch import TSAFusion

class NnupConvAct2ConvActPs(nn.Module):
    def __init__(self, in_ch=16, out_ch=12, deploy=False, upscale_factor=2):
        super(NnupConvAct2ConvActPs, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.deploy = deploy

        if deploy:
            self.up_rgb = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch * 4, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(upscale_factor)
            )
        else:
            self.up_rgb = nn.Sequential(
                nn.Upsample(scale_factor=upscale_factor, mode='nearest'),
                nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, in_fea):
        up_fea = self.up_rgb(in_fea)
        return up_fea

    def get_equivalent_kernel_bias(self):
        in_ch = self.up_rgb[1].in_channels
        out_ch = self.up_rgb[1].out_channels
        weight = self.up_rgb[1].weight
        bias = self.up_rgb[1].bias
        dst_out_ch = out_ch * 4

        weight_new = torch.zeros((dst_out_ch, in_ch, 3, 3)).to(weight.device)
        for ch in range(out_ch):
            weight_new[ch * 4, :, 0, 0] = weight[ch, :, 0, 0]
            weight_new[ch * 4, :, 0, 1] = weight[ch, :, 0, 1] + weight[ch, :, 0, 2]
            weight_new[ch * 4, :, 1, 0] = weight[ch, :, 1, 0] + weight[ch, :, 2, 0]
            weight_new[ch * 4, :, 1, 1] = weight[ch, :, 1, 1] + weight[ch, :, 1, 2] + \
                                          weight[ch, :, 2, 1] + weight[ch, :, 2, 2]

            weight_new[ch * 4 + 1, :, 0, 2] = weight[ch, :, 0, 2]
            weight_new[ch * 4 + 1, :, 0, 1] = weight[ch, :, 0, 0] + weight[ch, :, 0, 1]
            weight_new[ch * 4 + 1, :, 1, 2] = weight[ch, :, 1, 2] + weight[ch, :, 2, 2]
            weight_new[ch * 4 + 1, :, 1, 1] = weight[ch, :, 1, 0] + weight[ch, :, 1, 1] + \
                                              weight[ch, :, 2, 0] + weight[ch, :, 2, 1]

            weight_new[ch * 4 + 2, :, 2, 0] = weight[ch, :, 2, 0]
            weight_new[ch * 4 + 2, :, 1, 0] = weight[ch, :, 0, 0] + weight[ch, :, 1, 0]
            weight_new[ch * 4 + 2, :, 2, 1] = weight[ch, :, 2, 1] + weight[ch, :, 2, 2]
            weight_new[ch * 4 + 2, :, 1, 1] = weight[ch, :, 0, 1] + weight[ch, :, 0, 2] + \
                                              weight[ch, :, 1, 1] + weight[ch, :, 1, 2]

            weight_new[ch * 4 + 3, :, 2, 2] = weight[ch, :, 2, 2]
            weight_new[ch * 4 + 3, :, 1, 2] = weight[ch, :, 0, 2] + weight[ch, :, 1, 2]
            weight_new[ch * 4 + 3, :, 2, 1] = weight[ch, :, 2, 0] + weight[ch, :, 2, 1]
            weight_new[ch * 4 + 3, :, 1, 1] = weight[ch, :, 0, 0] + weight[ch, :, 0, 1] + \
                                              weight[ch, :, 1, 0] + weight[ch, :, 1, 1]

        bias_new = torch.zeros((dst_out_ch,)).to(weight.device)
        for ch in range(out_ch):
            bias_new[ch * 4: (ch + 1) * 4] = bias[ch]

        return weight_new, bias_new

    def switch_to_deploy(self, ):
        weight_new, bias_new = self.get_equivalent_kernel_bias()

        self.up_rgb = nn.Sequential(
            nn.Conv2d(in_channels=self.in_ch, out_channels=self.out_ch * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        self.up_rgb[0].weight.data = weight_new
        self.up_rgb[0].bias.data = bias_new

        self.deploy = True

# refer to DualBranchUnet_v48, for sunyue, less than 150ms
# @ARCH_REGISTRY.register()
class DualBranchUnet_v51(nn.Module):
    def __init__(self, **args):
        super(DualBranchUnet_v51, self).__init__()
        self.args = args

        self.head_conv = NnupConvAct2ConvActPs(in_ch=64, out_ch=16, deploy=args.get('deploy1', False))
        self.tail_conv = nn.Sequential()
        for ii in range(args.get('tail_conv', 1)):
            in_ch = 32 if 0 == ii else 16
            out_ch = 3 if args.get('tail_conv', 1) - 1 == ii else 16
            self.tail_conv.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1))
            self.tail_conv.append(nn.ReLU(inplace=True))

        if args.get('quad_bining', False):
            self.en_1_conv = nn.Sequential(
                nn.Conv2d(in_channels=4 * args.get('N_frame', 5) + args['quad_bining'], out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            )
        else:
            self.en_1_conv = nn.Sequential(
                nn.Conv2d(in_channels=4 * args.get('N_frame', 5), out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            )

        self.down_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        )

        self.down_2_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
        )

        self.down_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        )

        self.bottom_1_conv = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)

        self.up_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
        )

        self.decoder_3_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=128 + 64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        self.decoder_2_1_conv = nn.Sequential(
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, padding=0),
            # nn.ReLU(inplace=True),
            nn.Identity(),
        )

        self.up_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_1_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=64 + 32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Identity(),
        )

        self.up_rgb = nn.Sequential(
            NnupConvAct2ConvActPs(in_ch=64, out_ch=16, deploy=args.get('deploy2', False)),
            # nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        )

        self.lrelu = nn.ReLU(inplace=True)
        self.ret_list = args.get('ret_list', None)

    def forward(self, x):
        if self.args.get('quad_bining', False):
            x, bining = x[0], x[1]
            B, F, C, H, W = x.shape
            ft = x.reshape(B, -1, H, W)
            bining = bining.reshape(B, -1, H, W)
            ft = torch.cat([ft, bining], dim=1)
        else:
            B, F, C, H, W = x.shape
            ft = x.reshape(B, -1, H, W)

        features_en_1 = self.lrelu(self.en_1_conv(ft))
        features_down_1 = self.lrelu(self.down_1_conv(features_en_1))
        features_down_2 = self.lrelu(self.down_2_conv(features_down_1))
        features_down_3 = self.lrelu(self.down_3_conv(features_down_2))

        features_bottom = self.lrelu(self.bottom_1_conv(features_down_3))

        features_up_3 = self.up_3(features_bottom)
        features_de_3 = torch.cat([features_down_2, features_up_3], dim=1)
        features_de_3 = self.decoder_3_1_conv(features_de_3)

        features_up_2 = self.up_2(features_de_3)
        # features_de_2 = torch.cat([features_down_1, features_up_2], dim=1)
        features_de_2 = self.decoder_2_1_conv(features_up_2)

        features_up_1 = self.up_1(features_de_2)
        features_de_1 = torch.cat([features_en_1, features_up_1], dim=1)
        pre_rgb = self.decoder_1_1_conv(features_de_1)

        features_rgb = self.up_rgb(pre_rgb)

        # dm
        fea = self.head_conv(features_en_1)
        fea_dm = torch.cat([features_rgb, fea], dim=1)
        features_rgb = self.tail_conv(fea_dm)

        return features_rgb
