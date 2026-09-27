"""NeuralOperator 2.0.0 SpectralConv with one dtype-preserving allocation change.

The upstream forward implementation is otherwise verbatim. MIT license is in
NEURALOPERATOR-LICENSE. No installed package is changed.
"""
from typing import Optional, Tuple
import torch
from neuralop.layers.spectral_convolution import SpectralConv

UPSTREAM_SHA256 = "efd032b332833746f6021c3850ee1c1ad04420e95fdcbd676082b028fc9dc3b9"

class SpectralConvF64(SpectralConv):
    def forward(self, x: torch.Tensor, output_shape: Optional[Tuple[int]] = None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        if not self.complex_data:
            fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient in real spatial data
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        if self.complex_data:
            x = torch.fft.fftn(x, norm=self.fft_norm, dim=fft_dims)
            dims_to_fft_shift = fft_dims
        else:
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            # When x is real in spatial domain, the last half of the last dim is redundant.
            # See :ref:`fft_shift_explanation` for discussion of the FFT shift.
            dims_to_fft_shift = fft_dims[:-1]

        if self.order > 1:
            x = torch.fft.fftshift(x, dim=dims_to_fft_shift)

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = x.dtype  # Preserve float64 FFT as complex128.
        out_fft = torch.zeros(
            [batchsize, self.out_channels, *fft_size], device=x.device, dtype=out_dtype
        )

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        # if contraction is separable, weights have shape (channels, modes_x, ...)
        # otherwise they have shape (in_channels, out_channels, modes_x, ...)
        if self.separable:
            slices_w = [slice(None)]  # channels
        else:
            slices_w = [slice(None), slice(None)]  # in_channels, out_channels
        if self.complex_data:
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts
            ]
        else:
            # The last mode already has redundant half removed in real FFT
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]

        ### Pick the first n_modes modes of FFT signal along each dim

        # if separable conv, weight tensor only has one channel dim
        if self.separable:
            weight_start_idx = 1
        # otherwise drop first two dims (in_channels, out_channels)
        else:
            weight_start_idx = 2

        slices_x = [slice(None), slice(None)]  # Batch_size, channels

        for all_modes, kept_modes in zip(fft_size, list(weight.shape[weight_start_idx:])):
            # After fft-shift, the 0th frequency is located at n // 2 in each direction
            # We select n_modes modes around the 0th frequency (kept at index n//2) by grabbing indices
            # n//2 - n_modes//2  to  n//2 + n_modes//2       if n_modes is even
            # n//2 - n_modes//2  to  n//2 + n_modes//2 + 1   if n_modes is odd
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2

            # this slice represents the desired indices along each dim
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-1] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-1])
        else:
            slices_x[-1] = slice(None)

        slices_x = tuple(slices_x)
        out_fft[slices_x] = self._contract(
            x[slices_x], weight, separable=self.separable
        )

        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])

        if self.complex_data:
            x = torch.fft.ifftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
        else:
            x = torch.fft.irfftn(
                out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm
            )

        if self.bias is not None:
            x = x + self.bias

        return x
