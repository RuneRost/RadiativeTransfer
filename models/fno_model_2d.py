import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, List


class SpectralConv2d(eqx.Module):
    real_weights: jax.Array
    imag_weights: jax.Array
    in_channels: int
    out_channels: int
    modes_x: int
    modes_y: int

    def __init__(
            self,
            in_channels,
            out_channels,
            modes_x,
            modes_y,
            *,
            key,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y


        scale = 1.0 / (in_channels * out_channels)                      # check if this might need change

        real_key, imag_key = jax.random.split(key)
        self.real_weights = jax.random.uniform(
            real_key,
            (in_channels, out_channels, modes_x, modes_y),
            minval=-scale,
            maxval=+scale,
        )
        self.imag_weights = jax.random.uniform(
            imag_key,
            (in_channels, out_channels, modes_x, modes_y),
            minval=-scale,
            maxval=+scale,
        )    
    def complex_mult2d(
            self,
            x_hat,
            w,
    ):
        return jnp.einsum("iXY,ioXY->oXY", x_hat, w)  
    
    def __call__(
            self,
            x,
    ):
        channels, spatial_points_x, spatial_points_y = x.shape

        x_hat = jnp.fft.rfft2(x)                                            # ergänzen das axis 1 und 2?
        x_hat_under_modes = x_hat[:, :self.modes_x, :self.modes_y]
        weights = self.real_weights + 1j * self.imag_weights
        out_hat_under_modes = self.complex_mult2d(x_hat_under_modes, weights)

        out_hat = jnp.zeros(
            (self.out_channels, *x_hat.shape[1:]),
            dtype=x_hat.dtype
        )
        
        out_hat = out_hat.at[:, :self.modes_x, :self.modes_y].set(out_hat_under_modes)

        out = jnp.fft.irfft2(out_hat, s=[spatial_points_x, spatial_points_y])     #previously: n=spatial_points)

        return out
        #check if calculations are correct or any extra conditions as in create_turbulent_2D.py (Hermitian symmetry, ...)
 


class FNOBlock2d(eqx.Module):
    spectral_conv: SpectralConv2d
    bypass_conv: eqx.nn.Conv2d
    activation: Callable

    def __init__(
            self,
            in_channels,
            out_channels,
            modes_x,
            modes_y,
            activation,
            *,
            key,
    ):
        spectral_conv_key, bypass_conv_key = jax.random.split(key)
        self.spectral_conv = SpectralConv2d(
            in_channels,
            out_channels,
            modes_x,   
            modes_y,
            key=spectral_conv_key,
        )

        self.bypass_conv = eqx.nn.Conv2d(
            in_channels,
            out_channels,
            1,  # Kernel size is one
            key=bypass_conv_key,
        )
        self.activation = activation

    def __call__(
            self,
            x,
    ):
        return self.activation(
            self.spectral_conv(x) + self.bypass_conv(x)
        )
    

class FNO2d(eqx.Module):
    lifting: eqx.nn.Conv2d
    fno_blocks: List[FNOBlock2d]
    dropouts: List[eqx.nn.Dropout]
    projection: eqx.nn.Conv2d

    def __init__(
            self,
            in_channels,
            out_channels,
            modes_x,
            modes_y,
            width,
            p_do,
            activation,
            n_blocks = 4,
            *,
            key,
    ):
        key, lifting_key = jax.random.split(key)
        #lifting erhöht channel dim aber nicht spatial dim
        self.lifting = eqx.nn.Conv2d(
            in_channels,
            width,
            1,
            key=lifting_key,
        )

        self.fno_blocks = []
        self.dropouts = []
        for i in range(n_blocks):
            key, subkey = jax.random.split(key)  #bedeutet das, jeder Block wird gleich initialisiert, weil immer gleicher key?
            self.fno_blocks.append(FNOBlock2d(
                width,
                width,
                modes_x,
                modes_y,
                activation,
                key=subkey,
            ))
            self.dropouts.append(eqx.nn.Dropout(p=p_do))   
        #projection umgekehrt zu lifting
        key, projection_key = jax.random.split(key)
        self.projection = eqx.nn.Conv2d(
            width,
            out_channels,
            1,
            key=projection_key,
        )
        




    def __call__(
            self,
            x,
            key,
            deterministic: bool = False
    ):
        x = self.lifting(x)

        #for fno_block in self.fno_blocks:
        #    x = fno_block(x)

        keys = jax.random.split(key, len(self.dropouts))
        for i, fno_block in enumerate(self.fno_blocks):
            x = fno_block(x)
            if not deterministic:
                x = self.dropouts[i](x, key = keys[i])   # change this because now same key for every dropout and also same as for other type of layer (self.key doesnt get changes right?)

        x = self.projection(x)

        return x