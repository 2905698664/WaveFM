import torch
from torch import nn
import pywt
from typing import Sequence, Tuple, Union, List
from einops import rearrange, repeat
import torch.nn.functional as F

def _as_wavelet(wavelet):
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet) #
    else:
        return wavelet

def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
 
    a_flat = torch.reshape(a, [-1]) #
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)#
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul 

def construct_2d_filt(lo, hi) -> torch.Tensor:
    
    ll = _outer(lo, lo)
    lh = _outer(hi, lo)
    hl = _outer(lo, hi)
    hh = _outer(hi, hi)
    filt = torch.stack([ll, lh, hl, hh], 0) #[4,h,w]
    # filt = filt.unsqueeze(1) #[4,1,h,w]
    return filt

def get_filter_tensors(
        wavelet,
        flip: bool,
        device: Union[torch.device, str] = 'cpu',
        dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
   
    wavelet = _as_wavelet(wavelet)

    def _create_tensor(filter: Sequence[float]) -> torch.Tensor:
        if flip:
            if isinstance(filter, torch.Tensor):
                return filter.flip(-1).unsqueeze(0).to(device)
            else:
                return torch.tensor(filter[::-1], device=device, dtype=dtype).unsqueeze(0)
        else:
            if isinstance(filter, torch.Tensor):
                return filter.unsqueeze(0).to(device)
            else:
                return torch.tensor(filter, device=device, dtype=dtype).unsqueeze(0)

    dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank 
    dec_lo_tensor = _create_tensor(dec_lo)
    dec_hi_tensor = _create_tensor(dec_hi)
    rec_lo_tensor = _create_tensor(rec_lo)
    rec_hi_tensor = _create_tensor(rec_hi)
    return dec_lo_tensor, dec_hi_tensor, rec_lo_tensor, rec_hi_tensor


def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl


def fwt_pad2(
        data: torch.Tensor, wavelet, mode: str = "replicate" 
) -> torch.Tensor:
    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], len(wavelet.dec_lo)) 
    padr, padl = _get_pad(data.shape[-1], len(wavelet.dec_lo)) 

    data_pad = F.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad 

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        x = rearrange(x, 'b (g f) h w -> b g f h w', g=self.groups) #c= g*f
        x = rearrange(x, 'b g f h w -> b f g h w')
        x = rearrange(x, 'b f g h w -> b (f g) h w')
        return x

# global count
# count = 1
class LWN(nn.Module):
    def __init__(self, dim, wavelet='haar', initialize=True, head=4, drop_rate=0., use_ca=False, use_sa=False):
        super(LWN, self).__init__()
        self.dim = dim
        self.wavelet = _as_wavelet(wavelet)
        dec_lo, dec_hi, rec_lo, rec_hi = get_filter_tensors(
            wavelet, flip=True
        )# [1, filter_length]
        if initialize:
            self.dec_lo = nn.Parameter(dec_lo, requires_grad=True)
            self.dec_hi = nn.Parameter(dec_hi, requires_grad=True)
            self.rec_lo = nn.Parameter(rec_lo.flip(-1), requires_grad=True)
            self.rec_hi = nn.Parameter(rec_hi.flip(-1), requires_grad=True)
        else:
            self.dec_lo = nn.Parameter(torch.rand_like(dec_lo) * 2 - 1, requires_grad=True)
            self.dec_hi = nn.Parameter(torch.rand_like(dec_hi) * 2 - 1, requires_grad=True)
            self.rec_lo = nn.Parameter(torch.rand_like(rec_lo) * 2 - 1, requires_grad=True)
            self.rec_hi = nn.Parameter(torch.rand_like(rec_hi) * 2 - 1, requires_grad=True)

        self.wavedec = DWT(self.dec_lo, self.dec_hi, wavelet=wavelet, level=1)
        self.waverec = IDWT(self.rec_lo, self.rec_hi, wavelet=wavelet, level=1)

        self.conv1 = nn.Conv2d(dim*4, dim*6, 1)
        self.conv2 = nn.Conv2d(dim*6, dim*6, 7, padding=3, groups=dim*6)  # dw
        self.act = nn.GELU()
        self.conv3 = nn.Conv2d(dim*6, dim*4, 1)
        self.use_sa = use_sa
        self.use_ca = use_ca
        if self.use_sa:
            self.sa_h = nn.Sequential(
                nn.PixelShuffle(2), 
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)  # c -> 1
            )
            self.sa_v = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(dim // 4, 1, kernel_size=1, padding=0, stride=1, bias=True)
            )
            # self.sa_norm = LayerNorm2d(dim)
        if self.use_ca:
            self.ca_h = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True),  # conv2d
            )
            self.ca_v = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(dim, dim, 1, padding=0, stride=1, groups=1, bias=True)
            )
            self.shuffle = ShuffleBlock(2)

    def forward(self, x):
        _, _, H, W = x.shape
        ya, (yh, yv, yd) = self.wavedec(x)
        dec_x = torch.cat([ya, yh, yv, yd], dim=1)
        x = self.conv1(dec_x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        ya, yh, yv, yd = torch.chunk(x, 4, dim=1)
        y = self.waverec([ya, (yh, yv, yd)], None)
        if self.use_sa:
            sa_yh = self.sa_h(yh)
            sa_yv = self.sa_v(yv)
            y = y * (sa_yv + sa_yh)
        if self.use_ca:
            yh = torch.nn.functional.interpolate(yh, scale_factor=2, mode='area')
            yv = torch.nn.functional.interpolate(yv, scale_factor=2, mode='area')
            ca_yh = self.ca_h(yh)
            ca_yv = self.ca_v(yv)
            ca = self.shuffle(torch.cat([ca_yv, ca_yh], 1))  # channel shuffle
            ca_1, ca_2 = ca.chunk(2, dim=1)
            ca = ca_1 * ca_2   # gated channel attention
            y = y * ca
        return y

    def get_wavelet_loss(self):
        return self.perfect_reconstruction_loss()[0] + self.alias_cancellation_loss()[0]

    def perfect_reconstruction_loss(self):
        # polynomial multiplication is convolution, compute p(z):
        # print(dec_lo.shape, rec_lo.shape)
        pad = self.dec_lo.shape[-1] - 1
        p_lo = F.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0), # [1, 1, filter_length]
            self.rec_lo.flip(-1).unsqueeze(0), # [1, 1, filter_length]
            padding=pad)
        pad = self.dec_hi.shape[-1] - 1
        p_hi = F.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0), # [1, 1, filter_length]
            self.rec_hi.flip(-1).unsqueeze(0), # [1, 1, filter_length]
            padding=pad)

        p_test = p_lo + p_hi

        two_at_power_zero = torch.zeros(p_test.shape, device=p_test.device,
                                        dtype=p_test.dtype)
        two_at_power_zero[..., p_test.shape[-1] // 2] = 2
        # square the error
        errs = (p_test - two_at_power_zero) * (p_test - two_at_power_zero)
        return torch.sum(errs), p_test, two_at_power_zero#

    def alias_cancellation_loss(self):
        """ Implementation of the ac-loss as described on page 104 of Strang+Nguyen.
            F0(z)H0(-z) + F1(z)H1(-z) = 0 """
        m1 = torch.tensor([-1], device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        length = self.dec_lo.shape[-1]
        mask = torch.tensor([torch.pow(m1, n) for n in range(length)][::-1],
                            device=self.dec_lo.device, dtype=self.dec_lo.dtype)
        # polynomial multiplication is convolution, compute p(z):
        pad = self.dec_lo.shape[-1] - 1
        p_lo = torch.nn.functional.conv1d(
            self.dec_lo.flip(-1).unsqueeze(0) * mask,
            self.rec_lo.flip(-1).unsqueeze(0),
            padding=pad)

        pad = self.dec_hi.shape[-1] - 1
        p_hi = torch.nn.functional.conv1d(
            self.dec_hi.flip(-1).unsqueeze(0) * mask,
            self.rec_hi.flip(-1).unsqueeze(0),
            padding=pad)

        p_test = p_lo + p_hi
        zeros = torch.zeros(p_test.shape, device=p_test.device,
                            dtype=p_test.dtype)
        errs = (p_test - zeros) * (p_test - zeros)
        return torch.sum(errs), p_test, zeros


class DWT(nn.Module):
    def __init__(self, dec_lo, dec_hi, wavelet='haar', level=1, mode="replicate"):
        super(DWT, self).__init__()
        self.wavelet = _as_wavelet(wavelet)
        self.dec_lo = dec_lo
        self.dec_hi = dec_hi

        # # initial dec conv
        # self.conv = torch.nn.Conv2d(c1, c2 * 4, kernel_size=dec_filt.shape[-2:], groups=c1, stride=2)
        # self.conv.weight.data = dec_filt
        self.level = level
        self.mode = mode

    def forward(self, x):
        b, c, h, w = x.shape
        if self.level is None:
            self.level = pywt.dwtn_max_level([h, w], self.wavelet)
        wavelet_component: List[
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = [] 

        l_component = x
        dwt_kernel = construct_2d_filt(lo=self.dec_lo, hi=self.dec_hi)
        dwt_kernel = dwt_kernel.repeat(c, 1, 1)
        dwt_kernel = dwt_kernel.unsqueeze(dim=1)#
        for _ in range(self.level):
            l_component = fwt_pad2(l_component, self.wavelet, mode=self.mode)
            h_component = F.conv2d(l_component, dwt_kernel, stride=2, groups=c)
            res = rearrange(h_component, 'b (c f) h w -> b c f h w', f=4)
            l_component, lh_component, hl_component, hh_component = res.split(1, 2)
            wavelet_component.append((lh_component.squeeze(2), hl_component.squeeze(2), hh_component.squeeze(2)))
        wavelet_component.append(l_component.squeeze(2))
        return wavelet_component[::-1]


class IDWT(nn.Module):
    def __init__(self, rec_lo, rec_hi, wavelet='haar', level=1, mode="constant"):
        super(IDWT, self).__init__()
        self.rec_lo = rec_lo
        self.rec_hi = rec_hi
        self.wavelet = wavelet
        # self.convT = nn.ConvTranspose2d(c2 * 4, c1, kernel_size=weight.shape[-2:], groups=c1, stride=2)
        # self.convT.weight = torch.nn.Parameter(rec_filt)
        self.level = level
        self.mode = mode

    def forward(self, x, weight=None):
        l_component = x[0]
        _, c, _, _ = l_component.shape
        if weight is None:  # soft orthogonal
            idwt_kernel = construct_2d_filt(lo=self.rec_lo, hi=self.rec_hi)
            idwt_kernel = idwt_kernel.repeat(c, 1, 1)
            idwt_kernel = idwt_kernel.unsqueeze(dim=1)
        else:  # hard orthogonal
            idwt_kernel= torch.flip(weight, dims=[-1, -2])

        self.filt_len = idwt_kernel.shape[-1]
        for c_pos, component_lh_hl_hh in enumerate(x[1:]):
            l_component = torch.cat(
                # ll, lh, hl, hl, hh
                [l_component.unsqueeze(2), component_lh_hl_hh[0].unsqueeze(2),
                 component_lh_hl_hh[1].unsqueeze(2), component_lh_hl_hh[2].unsqueeze(2)], 2
            )
            # cat is not work for the strange transpose
            l_component = rearrange(l_component, 'b c f h w -> b (c f) h w')
            l_component = F.conv_transpose2d(l_component, idwt_kernel, stride=2, groups=c)

            # remove the padding
            padl = (2 * self.filt_len - 3) // 2
            padr = (2 * self.filt_len - 3) // 2
            padt = (2 * self.filt_len - 3) // 2
            padb = (2 * self.filt_len - 3) // 2
            if c_pos < len(x) - 2:
                pred_len = l_component.shape[-1] - (padl + padr)
                next_len = x[c_pos + 2][0].shape[-1]
                pred_len2 = l_component.shape[-2] - (padt + padb)
                next_len2 = x[c_pos + 2][0].shape[-2]
                if next_len != pred_len:
                    padr += 1
                    pred_len = l_component.shape[-1] - (padl + padr)
                    assert (
                            next_len == pred_len
                    ), "padding error, please open an issue on github "
                if next_len2 != pred_len2:
                    padb += 1
                    pred_len2 = l_component.shape[-2] - (padt + padb)
                    assert (
                            next_len2 == pred_len2
                    ), "padding error, please open an issue on github "
            if padt > 0:
                l_component = l_component[..., padt:, :]
            if padb > 0:
                l_component = l_component[..., :-padb, :]
            if padl > 0:
                l_component = l_component[..., padl:]
            if padr > 0:
                l_component = l_component[..., :-padr]
        return l_component

if __name__ == '__main__':
 
    x = torch.randn(1, 30, 64, 64)


    lwn = LWN(dim=30)

   
    output = lwn(x)
    print('output shape:', output.shape)
    print(lwn.get_wavelet_loss())
  