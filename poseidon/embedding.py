"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math

import numpy as np
import torch
import torch.nn as nn




class FourierExpansion(nn.Module):
    """A Fourier series-style expansion into a high-dimensional space.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    def __init__(self, lower: float, upper: float, d: int, assert_range: bool = True) -> None:
        """Initialise.

        Args:
            lower (float): Lower wavelength.
            upper (float): Upper wavelength.
            assert_range (bool, optional): Assert that the encoded tensor is within the specified
                wavelength range. Defaults to `True`.
        """
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range
        self.wavelengths = nn.Parameter(
            torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            d // 2,
            base=10,
        ), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (:class:`torch.Tensor`): Input to expand of shape `(..., n)`. All elements of `x` must
                lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d (int): Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x` are not
                within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            torch.Tensor: Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(self.lower <= x.abs(), torch.all(x.abs() <= self.upper))
        in_range_or_zero = torch.all(
            torch.logical_or(in_range, x == 0)
        )  # Allow zeros to pass through.
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range"
                f" `[{self.lower}, {self.upper}]`."
            )

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / self.wavelengths)
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.


if __name__=="__main__":
    levels_expansion = FourierExpansion(1, 30, 8)
    levels = torch.arange(1,31)
    print(levels)
    print(levels_expansion(levels.unsqueeze(0))[0,0])
    print(levels_expansion(levels.unsqueeze(0))[0,1])