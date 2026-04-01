import torch
import torch.nn as nn
from .wfft import rfftn,irfftn
import wml.wtorch.nn as wnn
from .conv_module import ConvModule


class FourierUnit(nn.Module):
    
    def __init__(self, in_channels, out_channels=None, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',act="ReLU"):
        super(FourierUnit, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2,
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        #self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.act = wnn.get_activation(act,inplace=True)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode

    @torch.cuda.amp.autocast(False)
    def forward(self, x):
        batch = x.shape[0]
        x = x.float()


        abs_dim = range(x.dim())
        fft_dim = (-2, -1)
        fft_dim = [abs_dim[d] for d in fft_dim]

        ffted = torch.fft.rfftn(x, dim=fft_dim)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])


        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.act(ffted)

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-2:]
        #output = irfftn(ffted, sl=ifft_shape_slice, dim=fft_dim)
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice,dim=fft_dim)


        return output

class HFFConv(nn.Module):
    def __init__(self,in_channels,conv_cfg=None,**kwargs):
        super().__init__()
        self.in_channels0 = in_channels//2
        self.in_channels1 = in_channels-self.in_channels0
        self.ffc_unit = FourierUnit(self.in_channels0,**kwargs)
        if conv_cfg is not None:
            self.conv = ConvModule(in_channels=self.in_channels1,out_channels=self.in_channels1,**conv_cfg)
        else:
            self.conv = None
    
    def forward(self,x):
        x0,x1 = torch.split(x,[self.in_channels0,self.in_channels1],dim=1)
        x0 = self.ffc_unit(x0)
        if self.conv is not None:
            x1 = self.conv(x1)
        return torch.cat([x0,x1],dim=1)
