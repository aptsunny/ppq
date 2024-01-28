import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLayer(nn.Module):

    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

class NearestNeighborUpsample(nn.Module):

    def __init__(self, scale_factor=2):
        super(NearestNeighborUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # 获取输入的形状
        batch_size, channels, in_height, in_width = x.size()

        # 计算输出的尺寸
        out_height = in_height * self.scale_factor
        out_width = in_width * self.scale_factor

        # 使用torch.nn.functional.interpolate进行最近邻插值
        output = F.interpolate(x, size=(out_height, out_width), mode='nearest')

        return output

class NaiveNearestNeighborUpsample(nn.Module):

    def __init__(self, scale_factor=2):
        super(NaiveNearestNeighborUpsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        out_height = in_height * self.scale_factor
        out_width = in_width * self.scale_factor

        # 创建一个与输出尺寸匹配的网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(out_height), torch.arange(out_width))
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).float() / self.scale_factor
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).float() / self.scale_factor

        # 将网格坐标转换为输入特征图上的索引
        indices_y = (grid_y + 0.5).clamp(
            min=0, max=in_height - 1).long().cuda()
        indices_x = (grid_x + 0.5).clamp(min=0, max=in_width - 1).long().cuda()

        # 使用gather函数根据索引从输入特征图中取值
        output = x.reshape(batch_size, channels, -1)
        output = output.gather(
            dim=-1, index=torch.stack([indices_y, indices_x],
                                      dim=-1).view(-1)).reshape(
                                          batch_size, channels, out_height,
                                          out_width)

        return output

class NaiveNearestNeighborUpsample_V2(nn.Module):

    def __init__(self, scale_factor=2):
        super(NaiveNearestNeighborUpsample_V2, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, channels, in_height, in_width = x.size()
        out_height = in_height * self.scale_factor
        out_width = in_width * self.scale_factor

        # 创建一个与输出尺寸匹配的网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(out_height), torch.arange(out_width))
        grid_y = grid_y.unsqueeze(0).float() / self.scale_factor
        grid_x = grid_x.unsqueeze(0).float() / self.scale_factor

        # 将网格坐标转换为输入特征图上的索引
        indices_y = (grid_y + 0.5).clamp(
            min=0, max=in_height - 1).long().cuda()
        indices_x = (grid_x + 0.5).clamp(min=0, max=in_width - 1).long().cuda()

        # 将一维化的索引展平并重塑为(Batch, Channels, Out_Height*Out_Width)形式
        indices_y = indices_y.reshape(1, 1, out_height * out_width)
        indices_x = indices_x.reshape(1, 1, out_height * out_width)

        # 将输入特征图展平为(Batch, Channels, In_Height*In_Width)形式
        x_flattened = x.reshape(batch_size, channels, -1)

        # 使用gather函数根据索引从输入特征图中取值
        output = x_flattened.gather(
            dim=-1, index=torch.stack([indices_y, indices_x],
                                      dim=-1).view(-1)).reshape(
                                          batch_size, channels, out_height,
                                          out_width)

        return output

class NaiveNearestNeighborUpsample_V3(nn.Module):

    def __init__(self, scale_factor=2):
        super(NaiveNearestNeighborUpsample_V3, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, input):
        batch_size, channels, in_height, in_width = input.size()
        out_height = in_height * self.scale_factor
        out_width = in_width * self.scale_factor

        # 创建输出张量并填充
        output = torch.zeros(batch_size, channels, out_height,
                             out_width).cuda()

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        y = i // self.scale_factor
                        x = j // self.scale_factor
                        output[b, c, i, j] = input[b, c, y, x]

        return output

class Upsample_expand(nn.Module):

    def __init__(self, stride=2):
        super(Upsample_expand, self).__init__()
        self.stride = stride

    def forward(self, x):
        assert (x.data.dim() == 4)

        x = x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
            expand(x.size(0), x.size(1), x.size(2), self.stride, x.size(3), self.stride).contiguous().\
            view(x.size(0), x.size(1), x.size(2) * self.stride, x.size(3) * self.stride)

        return x

class NnupConvAct2ConvActPs(nn.Module):

    def __init__(self, in_ch=16, out_ch=12, deploy=False):
        super(NnupConvAct2ConvActPs, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.deploy = deploy
        # self.id_layer = NearestNeighborUpsample()
        # self.id_layer = NaiveNearestNeighborUpsample()
        # self.id_layer = NaiveNearestNeighborUpsample_V2()
        # self.id_layer = NaiveNearestNeighborUpsample_V3()
        # self.id_layer = IdentityLayer()
        self.id_layer = Upsample_expand()

        if deploy:
            self.up_rgb = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch * 4,
                    kernel_size=3,
                    padding=1), nn.ReLU(inplace=True), nn.PixelShuffle(2))
        else:
            self.up_rgb = nn.Sequential(
                # nn.Upsample(scale_factor=2, mode='nearest'),
                self.id_layer,
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=1),
                nn.ReLU(inplace=True))

    def forward(self, in_fea):
        # in_fea = F.interpolate(in_fea, scale_factor=2, mode='nearest')
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
            weight_new[ch * 4, :, 0,
                       1] = weight[ch, :, 0, 1] + weight[ch, :, 0, 2]
            weight_new[ch * 4, :, 1,
                       0] = weight[ch, :, 1, 0] + weight[ch, :, 2, 0]
            weight_new[ch * 4, :, 1, 1] = weight[ch, :, 1, 1] + weight[ch, :, 1, 2] + \
                                          weight[ch, :, 2, 1] + weight[ch, :, 2, 2]

            weight_new[ch * 4 + 1, :, 0, 2] = weight[ch, :, 0, 2]
            weight_new[ch * 4 + 1, :, 0,
                       1] = weight[ch, :, 0, 0] + weight[ch, :, 0, 1]
            weight_new[ch * 4 + 1, :, 1,
                       2] = weight[ch, :, 1, 2] + weight[ch, :, 2, 2]
            weight_new[ch * 4 + 1, :, 1, 1] = weight[ch, :, 1, 0] + weight[ch, :, 1, 1] + \
                                              weight[ch, :, 2, 0] + weight[ch, :, 2, 1]

            weight_new[ch * 4 + 2, :, 2, 0] = weight[ch, :, 2, 0]
            weight_new[ch * 4 + 2, :, 1,
                       0] = weight[ch, :, 0, 0] + weight[ch, :, 1, 0]
            weight_new[ch * 4 + 2, :, 2,
                       1] = weight[ch, :, 2, 1] + weight[ch, :, 2, 2]
            weight_new[ch * 4 + 2, :, 1, 1] = weight[ch, :, 0, 1] + weight[ch, :, 0, 2] + \
                                              weight[ch, :, 1, 1] + weight[ch, :, 1, 2]

            weight_new[ch * 4 + 3, :, 2, 2] = weight[ch, :, 2, 2]
            weight_new[ch * 4 + 3, :, 1,
                       2] = weight[ch, :, 0, 2] + weight[ch, :, 1, 2]
            weight_new[ch * 4 + 3, :, 2,
                       1] = weight[ch, :, 2, 0] + weight[ch, :, 2, 1]
            weight_new[ch * 4 + 3, :, 1, 1] = weight[ch, :, 0, 0] + weight[ch, :, 0, 1] + \
                                              weight[ch, :, 1, 0] + weight[ch, :, 1, 1]

        bias_new = torch.zeros((dst_out_ch, )).to(weight.device)
        for ch in range(out_ch):
            bias_new[ch * 4:(ch + 1) * 4] = bias[ch]

        return weight_new, bias_new

    def switch_to_deploy(self, ):
        weight_new, bias_new = self.get_equivalent_kernel_bias()

        self.up_rgb = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_ch,
                out_channels=self.out_ch * 4,
                kernel_size=3,
                padding=1), nn.ReLU(inplace=True), nn.PixelShuffle(2))
        self.up_rgb[0].weight.data = weight_new
        self.up_rgb[0].bias.data = bias_new

        self.deploy = True

# refer to DualBranchUnet_v36, add relu after up_3
# @ARCH_REGISTRY.register()
class DualBranchUnet_v43(nn.Module):

    def __init__(self, **args):
        super(DualBranchUnet_v43, self).__init__()
        self.args = args

        self.head_conv = NnupConvAct2ConvActPs(
            in_ch=64, out_ch=16, deploy=args.get('deploy1', False))
        self.tail_conv = nn.Conv2d(
            in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.en_1_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=4 * args.get('N_frame', 5),
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1))

        self.down_1_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
        )

        self.down_2_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1),
        )

        self.down_3_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1),
        )

        self.bottom_1_conv = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.up_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=128 * 4,
                kernel_size=3,
                padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_3_1_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=256 + 128,
                out_channels=256,
                kernel_size=1,
                padding=0),
            nn.ReLU(inplace=True),
        )

        self.up_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=64 * 4, kernel_size=3,
                padding=1), nn.ReLU(inplace=True), nn.PixelShuffle(2))
        self.decoder_2_1_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=128 + 64,
                out_channels=128,
                kernel_size=1,
                padding=0),
            nn.ReLU(inplace=True),
        )

        self.up_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=32 * 4, kernel_size=3,
                padding=1), nn.ReLU(inplace=True), nn.PixelShuffle(2))
        self.decoder_1_1_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=64 + 32, out_channels=64, kernel_size=3,
                padding=1), nn.ReLU(inplace=True))

        self.up_rgb = nn.Sequential(
            NnupConvAct2ConvActPs(
                in_ch=64, out_ch=16, deploy=args.get('deploy2', False)),
            nn.Conv2d(
                in_channels=16, out_channels=3, kernel_size=3, padding=1))

        self.lrelu = nn.ReLU(inplace=True)
        self.ret_list = args.get('ret_list', None)

    def forward(self, x):
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
        features_de_2 = torch.cat([features_down_1, features_up_2], dim=1)
        features_de_2 = self.decoder_2_1_conv(features_de_2)

        features_up_1 = self.up_1(features_de_2)
        features_de_1 = torch.cat([features_en_1, features_up_1], dim=1)
        pre_rgb = self.decoder_1_1_conv(features_de_1)  # + base_frame

        features_rgb = self.up_rgb(pre_rgb)

        # dm
        # base_frame = ft[:, :4, :, :]
        fea = self.head_conv(features_en_1)
        dm_3c = self.tail_conv(fea)

        features_rgb = self.lrelu(dm_3c + features_rgb)
        # features_rgb = features_rgb + dm_3c

        return features_rgb

# refer to DualBranchUnet_v43
class DualBranchUnet_v43_addConv1(nn.Module):
    def __init__(self, **args):
        super(DualBranchUnet_v43_addConv1, self).__init__()
        self.args = args

        self.head_conv = NnupConvAct2ConvActPs(in_ch=64, out_ch=16, deploy=args.get('deploy1', False))
        self.tail_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.en_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=4*args.get('N_frame', 5), out_channels=32, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        )

        self.down_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        )

        self.bottom_1_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.up_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_3_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=256 + 128, out_channels=256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.up_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        self.decoder_2_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=128 + 64, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.up_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        self.decoder_1_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=64 + 32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_rgb = nn.Sequential(
            NnupConvAct2ConvActPs(in_ch=64, out_ch=16, deploy=args.get('deploy2', False)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        )

        self.after_rgb = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.lrelu = nn.ReLU(inplace=True)
        self.ret_list = args.get('ret_list', None)

    def forward(self, x):
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
        features_de_2 = torch.cat([features_down_1, features_up_2], dim=1)
        features_de_2 = self.decoder_2_1_conv(features_de_2)

        features_up_1 = self.up_1(features_de_2)
        features_de_1 = torch.cat([features_en_1, features_up_1], dim=1)
        pre_rgb = self.decoder_1_1_conv(features_de_1)# + base_frame

        features_rgb = self.up_rgb(pre_rgb)

        # dm
        # base_frame = ft[:, :4, :, :]
        fea = self.head_conv(features_en_1)
        dm_3c = self.tail_conv(fea)

        features_rgb = self.lrelu(dm_3c + features_rgb)
        features_rgb = self.after_rgb(features_rgb)

        return features_rgb

# refer to DualBranchUnet_v43
class DualBranchUnet_v43_addConv2(nn.Module):
    def __init__(self, **args):
        super(DualBranchUnet_v43_addConv2, self).__init__()
        self.args = args

        self.head_conv = NnupConvAct2ConvActPs(in_ch=64, out_ch=16, deploy=args.get('deploy1', False))
        self.tail_conv = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)

        self.en_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=4*args.get('N_frame', 5), out_channels=32, kernel_size=3, stride=1, padding=1),
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
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        )

        self.down_3_conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
        )

        self.bottom_1_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.up_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder_3_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=256 + 128, out_channels=256, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.up_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        self.decoder_2_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=128 + 64, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.up_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32 * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2)
        )
        self.decoder_1_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=64 + 32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up_rgb = nn.Sequential(
            NnupConvAct2ConvActPs(in_ch=64, out_ch=16, deploy=args.get('deploy2', False)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        )

        self.after_rgb = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.lrelu = nn.ReLU(inplace=True)
        self.ret_list = args.get('ret_list', None)

    def forward(self, x):
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
        features_de_2 = torch.cat([features_down_1, features_up_2], dim=1)
        features_de_2 = self.decoder_2_1_conv(features_de_2)

        features_up_1 = self.up_1(features_de_2)
        features_de_1 = torch.cat([features_en_1, features_up_1], dim=1)
        pre_rgb = self.decoder_1_1_conv(features_de_1)# + base_frame

        features_rgb = self.up_rgb(pre_rgb)

        # dm
        # base_frame = ft[:, :4, :, :]
        fea = self.head_conv(features_en_1)
        dm_3c = self.tail_conv(fea)

        features_rgb = self.lrelu(dm_3c + features_rgb)
        features_rgb = self.after_rgb(features_rgb)

        return features_rgb
