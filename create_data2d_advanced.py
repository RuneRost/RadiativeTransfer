import jax
import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt
from create_data.create_turbulent_2D import generate_correlated_lognormal_field
from ray_tracing_many_stars import compute_radiation_field_from_star, compute_radiation_field_from_multiple_stars, gaussian_emissivity

def create_data_2d_advanced(key, Nx=200, Ny=200, num_rays=360,step_size=0.2, max_steps=1500):

    key = random.PRNGKey(42)

    Nx, Ny = 200, 200



    kappa, mask = generate_correlated_lognormal_field(key, shape=(Nx, Ny), mean=1.0, length_scale=0.05, sigma_g=1.2)


    # Extract coordinates of top 1% (True values in mask)
    star_indices = jnp.argwhere(mask)

    star_positions = star_indices.astype(jnp.float32) + 0.5  # (x, y) coords


    emissivity = jnp.zeros((Nx, Ny))

    for pos in star_positions:
        emissivity += gaussian_emissivity(Nx, Ny, center=pos, amplitude=1e3, width=5.0)

    J_multi = compute_radiation_field_from_multiple_stars(
        emissivity, kappa, star_positions,
        num_rays=360, step_size=0.2, max_steps=1500
    )

    return jnp.array([J_multi, kappa, mask, star_positions, emissivity])
    
create_data_2d_advanced_vmapped = vmap(create_data_2d_advanced, in_axes=(0, None, None, None, None, None))

create_data_2d_advanced_vmapped_jitted = jax.jit(create_data_2d_advanced_vmapped)


