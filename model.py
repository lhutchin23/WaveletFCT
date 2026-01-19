import torch
import torch.nn as nn
import torch.nn.functional as F


class MorletWaveletDecomp2D(nn.Module):
    """
    Morlet Wavelet Decompositon 2D
    Applies 1D morlet filters horizontally and vertically to create 4 subbands:
    LL (approximation)
    LH (Horizontal Details)
    HL (Vertical Details)
    HH (Diagonal Details)

    We use real_morlet instead of the complex morlet because we want the function to be symmetric about 0.

    Learnable parameters:
    - centre_freq: Centre frequency of morlet wavelet
    - bandwidth_freq: Bandwidth Parameter for the Morlet Wavelet (like width)
    - scales: Dilation for multi-resolution

    We do 2D decomp, meaning our output are 4 bands that are a downsample by a factor of 2.
    """

    def __init__(self, in_channels, kernel_size=16):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # log_based so that the parameters > 0 when we do e ^ (log) later.
        self.log_centre_freq = nn.Parameter(torch.log(torch.tensor(2.0)))
        self.log_bandwidth_freq = nn.Parameter(torch.log(torch.tensor(1.0)))

        self.register_buffer("t", torch.linspace(-4, 4, kernel_size))

        self.filter_LL = nn.Parameter(
            torch.zeros(in_channels, 1, kernel_size, kernel_size)
        )
        self.filter_LH = nn.Parameter(
            torch.zeros(in_channels, 1, kernel_size, kernel_size)
        )
        self.filter_HL = nn.Parameter(
            torch.zeros(in_channels, 1, kernel_size, kernel_size)
        )
        self.filter_HH = nn.Parameter(
            torch.zeros(in_channels, 1, kernel_size, kernel_size)
        )

        self._initialize_filters()

    # defines it so that they convert back to a positive value
    @property
    def centre_freq(self):
        return torch.exp(self.log_centre_freq)

    @property
    def bandwidth_freq(self):
        return torch.exp(self.log_bandwidth_freq)

    """
    the real-valued morlet wavelet basis is:
    1/(K * bandwith) * e^(-(bandwith * t)^2)*cos(2*pi*centre_freq*t)
    Since we normalize the addition of term 1/K doesn't impact our results since K is typically a constant associated with resolution
    """

    def _morlet_1d(self):
        centre_freq = self.centre_freq
        bandwidth_freq = self.bandwidth_freq

        gaussian = torch.exp(-((bandwidth_freq * self.t) ** 2))  # type: ignore
        wave_real = torch.cos(2 * torch.pi * centre_freq * self.t)  # type: ignore

        morlet_real = gaussian * wave_real

        # Normalize
        morlet_real = morlet_real / (morlet_real.abs().sum() + 1e-10)
        lowpass = gaussian / (gaussian.sum() + 1e-10)

        return lowpass, morlet_real

    def _initialize_filters(self):
        with torch.no_grad():
            lowpass, morlet_real = self._morlet_1d()

            LL_2d = torch.outer(lowpass, lowpass)
            LH_2d = torch.outer(lowpass, morlet_real)
            HL_2d = torch.outer(morlet_real, lowpass)
            HH_2d = torch.outer(morlet_real, morlet_real)

            # Broadcast to all channels
            self.filter_LL.data[:, 0, :, :] = LL_2d
            self.filter_LH.data[:, 0, :, :] = LH_2d
            self.filter_HL.data[:, 0, :, :] = HL_2d
            self.filter_HH.data[:, 0, :, :] = HH_2d

    def forward(self, x):
        """
        applies filters to yield our 4 subbands, we will use LL as our downsample since it represents
        an approximation of the image we feed it.
        We will use the details for upsampling, it preserves important details about the structure for later.
        """
        padding = (self.kernel_size - 2) // 2

        LL = F.conv2d(
            x, self.filter_LL, stride=2, padding=padding, groups=self.in_channels
        )
        LH = F.conv2d(
            x, self.filter_LH, stride=2, padding=padding, groups=self.in_channels
        )
        HL = F.conv2d(
            x, self.filter_HL, stride=2, padding=padding, groups=self.in_channels
        )
        HH = F.conv2d(
            x, self.filter_HH, stride=2, padding=padding, groups=self.in_channels
        )

        return LL, LH, HL, HH


class MorletInverseWavelet2D(nn.Module):
    """
    Inverse Wavelet Transformer; upsamples by a factor of 2.
    Reconstructs from a LL-subband that we modify, then
    with the addition of the LH,HL,HH subbands to upsample by a factor of 2x.
    """

    def __init__(self, in_channels, kernel_size=16):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.recon_LL = nn.Parameter(
            torch.ones(in_channels, 1, kernel_size, kernel_size)
        )
        self.recon_LH = nn.Parameter(
            torch.ones(in_channels, 1, kernel_size, kernel_size)
        )
        self.recon_HL = nn.Parameter(
            torch.ones(in_channels, 1, kernel_size, kernel_size)
        )
        self.recon_HH = nn.Parameter(
            torch.ones(in_channels, 1, kernel_size, kernel_size)
        )

        with torch.no_grad():
            # init filters with 1/K
            self.recon_LL.data.fill_(1.0 / self.kernel_size)
            self.recon_LH.data.fill_(1.0 / self.kernel_size)
            self.recon_HL.data.fill_(1.0 / self.kernel_size)
            self.recon_HH.data.fill_(1.0 / self.kernel_size)

    def forward(self, LL, LH, HL, HH):
        # returns upsampling by 2x using idwt
        padding = (self.kernel_size - 2) // 2  # upsamples
        recon_LL = F.conv_transpose2d(
            LL,
            self.recon_LL,
            stride=2,
            padding=padding,
            output_padding=0,
            groups=self.in_channels,
        )
        recon_LH = F.conv_transpose2d(
            LH,
            self.recon_LH,
            stride=2,
            padding=padding,
            output_padding=0,
            groups=self.in_channels,
        )
        recon_HL = F.conv_transpose2d(
            HL,
            self.recon_HL,
            stride=2,
            padding=padding,
            output_padding=0,
            groups=self.in_channels,
        )
        recon_HH = F.conv_transpose2d(
            HH,
            self.recon_HH,
            stride=2,
            padding=padding,
            output_padding=0,
            groups=self.in_channels,
        )
        recon = recon_LL + recon_LH + recon_HL + recon_HH
        return recon


class WaveletDenoiseBlock(nn.Module):
    """
    Morlet based denoising using learnable soft-thresholding

    Converts to 4 subbands with wavelet decomp, soft-thresholds the coefficients of the detail bands
    applies iDWT to reconstruct the signal with less noise

    parameter is a torch parameter so it should ideally adapt to level of noise found in the scans
    """

    def __init__(self, in_channels, kernel_size=16):
        super().__init__()
        self.dwt = MorletWaveletDecomp2D(in_channels, kernel_size)
        self.idwt = MorletInverseWavelet2D(in_channels, kernel_size)

        self.log_threshold = nn.Parameter(torch.log(torch.tensor(0.05)))

    @property
    def threshold(self):
        return torch.exp(self.log_threshold)

    def soft_threshhold(self, x, threshold):
        # applies standard soft_thresholding to our input
        magnitude = torch.abs(x)
        denoise = torch.sigmoid(10 * (magnitude - threshold))
        return x * denoise

    def forward(self, x):
        LL, LH, HL, HH = self.dwt(x)

        # soft denoising
        LH_denoise = self.soft_threshhold(LH, self.threshold)
        HL_denoise = self.soft_threshhold(HL, self.threshold)
        HH_denoise = self.soft_threshhold(HH, self.threshold)

        x_denoised = self.idwt(LL, LH_denoise, HL_denoise, HH_denoise)
        return x_denoised


# Implementation of FCT comes from the official FCT github (https://github.com/Thanos-DB/FullyConvolutionalTransformer)
class Attention(nn.Module):
    def __init__(
        self,
        channels,
        num_heads,
        proj_drop=0.0,
        kernel_size=3,
        stride_kv=1,
        stride_q=1,
        padding_kv="same",
        padding_q="same",
        attention_bias=True,
    ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride_q,
            padding_q,
            bias=attention_bias,
            groups=channels,
        )
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride_kv,
            stride_kv,
            bias=attention_bias,
            groups=channels,
        )
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            stride_kv,
            stride_kv,
            bias=attention_bias,
            groups=channels,
        )
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(
            embed_dim=channels, bias=attention_bias, batch_first=True, num_heads=1
        )
        self.dropout = nn.Dropout(proj_drop)

    def _build_projection(self, x, qkv):
        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj  # type: ignore

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        h = w = int(x1.shape[2] ** 0.5)
        x1 = x1.view(x1.shape[0], x1.shape[1], h, w)
        x1 = self.dropout(x1)
        return x1


class Transformer(nn.Module):

    def __init__(
        self,
        out_channels,
        num_heads,
        dpr,
        proj_drop=0.0,
        attention_bias=True,
        padding_q="same",
        padding_kv="same",
        stride_kv=1,
        stride_q=1,
    ):
        super().__init__()

        self.attention_output = Attention(
            channels=out_channels,
            num_heads=num_heads,
            proj_drop=proj_drop,
            padding_q=padding_q,
            padding_kv=padding_kv,
            stride_kv=stride_kv,
            stride_q=stride_q,
            attention_bias=attention_bias,
        )

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        return x3


class Wide_Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, 3, 1, padding="same", dilation=2
        )
        self.conv3 = nn.Conv2d(
            in_channels, out_channels, 3, 1, padding="same", dilation=3
        )
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x1 = self.dropout(F.gelu(self.conv1(x)))
        x2 = self.dropout(F.gelu(self.conv2(x)))
        x3 = self.dropout(F.gelu(self.conv3(x)))
        added = x1 + x2 + x3
        x_out = self.dropout(F.gelu(self.conv4(added)))
        return x_out


class Block_encoder_bottleneck(nn.Module):
    def __init__(
        self, blk, in_channels, out_channels, att_heads, dpr, wavelet_kernel=16
    ):
        super().__init__()
        self.blk = blk
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.dropout = nn.Dropout(0.3)

        if (self.blk == "first") or (self.blk == "bottleneck"):
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        else:
            self.conv1 = nn.Conv2d(1, in_channels, 3, 1, padding="same")
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
            self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")

        self.trans = Transformer(out_channels, att_heads, dpr)
        self.wavelet_pool = MorletWaveletDecomp2D(out_channels, kernel_size=wavelet_kernel)

    def forward(self, x, scale_img=None):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)

        if (self.blk == "first") or (self.blk == "bottleneck"):
            x1 = F.relu(self.conv1(x1))
            x1 = F.relu(self.conv2(x1))
        else:
            x1 = torch.cat((F.relu(self.conv1(scale_img)), x1), dim=1)
            x1 = F.relu(self.conv2(x1))
            x1 = F.relu(self.conv3(x1))

        x1 = self.dropout(x1)
        LL, LH, HL, HH = self.wavelet_pool(x1)
        out = self.trans(LL)
        return out, (LH, HL, HH)


class Block_decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        att_heads,
        dpr,
        wavelet_kernel=16,
    ):
        super().__init__()
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.wavelet_upsample = MorletInverseWavelet2D(in_channels, kernel_size=wavelet_kernel)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels * 2, out_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.trans = Transformer(out_channels, att_heads, dpr)
        self.dropout = nn.Dropout(0.3)

        self.weight_LH = nn.Parameter(torch.tensor(0.3))
        self.weight_HL = nn.Parameter(torch.tensor(0.3))
        self.weight_HH = nn.Parameter(torch.tensor(0.3))

    def forward(self, x, skip, encoder_details):
        x1 = x.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)

        LH, HL, HH = encoder_details
        w_lh = F.softplus(self.weight_LH)
        w_hl = F.softplus(self.weight_HL)
        w_hh = F.softplus(self.weight_HH)
        x1 = self.wavelet_upsample(x1, LH * w_lh, HL * w_hl, HH * w_hh)

        x1 = F.relu(self.conv1(x1))
        x1 = torch.cat((skip, x1), dim=1)
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = self.dropout(x1)
        out = self.trans(x1)
        return out


class DS_out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.layernorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.layernorm(x1)
        x1 = x1.permute(0, 3, 1, 2)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        out = self.conv3(x1)

        return out


class FCT(nn.Module):
    def __init__(
        self,
        num_classes=4,
        wavelet_kernel=16,
    ):
        super().__init__()

        att_heads = [2, 2, 2, 2, 2, 2, 2, 2, 2]
        filters = [8, 16, 32, 64, 128, 64, 32, 16, 8]

        blocks = len(filters)
        dpr = [0.0] * blocks

        self.denoise = WaveletDenoiseBlock(1, kernel_size=wavelet_kernel)

         
        # wavelet decomp instead of avg pool:wq
        self.scale_wavelet = MorletWaveletDecomp2D(1, kernel_size=wavelet_kernel)

        self.block_1 = Block_encoder_bottleneck(
            "first", 1, filters[0], att_heads[0], dpr[0], wavelet_kernel
        )
        self.block_2 = Block_encoder_bottleneck(
            "second", filters[0], filters[1], att_heads[1], dpr[1], wavelet_kernel
        )
        self.block_3 = Block_encoder_bottleneck(
            "third", filters[1], filters[2], att_heads[2], dpr[2], wavelet_kernel
        )
        self.block_4 = Block_encoder_bottleneck(
            "fourth", filters[2], filters[3], att_heads[3], dpr[3], wavelet_kernel
        )
        self.block_5 = Block_encoder_bottleneck(
            "bottleneck", filters[3], filters[4], att_heads[4], dpr[4], wavelet_kernel
        )

        #  idwt for upsampling
        self.block_6 = Block_decoder(
            filters[4],
            filters[5],
            att_heads[5],
            dpr[5],
            wavelet_kernel,
        )
        self.block_7 = Block_decoder(
            filters[5],
            filters[6],
            att_heads[6],
            dpr[6],
            wavelet_kernel,
        )
        self.block_8 = Block_decoder(
            filters[6],
            filters[7],
            att_heads[7],
            dpr[7],
            wavelet_kernel,
        )
        self.block_9 = Block_decoder(
            filters[7],
            filters[8],
            att_heads[8],
            dpr[8],
            wavelet_kernel,
        )

        self.ds7 = DS_out(filters[6], num_classes)
        self.ds8 = DS_out(filters[7], num_classes)
        self.ds9 = DS_out(filters[8], num_classes)

    def forward(self, x):
        x = self.denoise(x)

        scale_img_2, _, _, _ = self.scale_wavelet(x)
        scale_img_3, _, _, _ = self.scale_wavelet(scale_img_2)
        scale_img_4, _, _, _ = self.scale_wavelet(scale_img_3)

        x, _ = self.block_1(x)
        skip1 = x
        x, detail_2 = self.block_2(x, scale_img_2)
        skip2 = x
        x, detail_3 = self.block_3(x, scale_img_3)
        skip3 = x
        x, detail_4 = self.block_4(x, scale_img_4)
        skip4 = x
        x, detail_5 = self.block_5(x)

        x = self.block_6(x, skip4, detail_5)
        x = self.block_7(x, skip3, detail_4)
        skip7 = x
        x = self.block_8(x, skip2, detail_3)
        skip8 = x
        x = self.block_9(x, skip1, detail_2)
        skip9 = x

        out7 = self.ds7(skip7)
        out8 = self.ds8(skip8)
        out9 = self.ds9(skip9)

        return out7, out8, out9

    def get_learned_wavelet_params(self):
        params = {
            "centre_freq": self.block_1.wavelet_pool.centre_freq.item(),
            "bandwidth_freq": self.block_1.wavelet_pool.bandwidth_freq.item(),
        }
        params["denoise_threshold"] = self.denoise.threshold.item()
        return params


