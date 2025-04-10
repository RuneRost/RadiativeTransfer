import jax
import jax.numpy as jnp
from jax import random, vmap
import matplotlib.pyplot as plt
from create_data.create_turbulent_2D import generate_correlated_lognormal_field
#import os
#import time



def compute_radiation_field_from_star(j_map, kappa_map, source_pos, num_rays=360, step_size=0.5, max_steps=1000):
    """
    Emit rays from a point source in all directions and compute the radiation field J(x, y).

    Args:
        j_map: 2D emissivity field
        kappa_map: 2D absorption field
        source_pos: (x0, y0) position of the star
        num_rays: number of directions to shoot
        step_size: marching step
        max_steps: max steps per ray

    Returns:
        J: 2D array of accumulated intensity per cell
    """
    Nx, Ny = j_map.shape
    J = jnp.zeros((Nx, Ny))

    # Sample directions uniformly over 2π
    angles = jnp.linspace(0, 2 * jnp.pi, num_rays, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    def trace_and_accumulate(J, direction):
        def body_fn(i, state):
            x, y, I, tau, J = state

            # Clamp to grid
            ix = jnp.clip(jnp.floor(x).astype(int), 0, Nx - 1)
            iy = jnp.clip(jnp.floor(y).astype(int), 0, Ny - 1)

            j_val = j_map[ix, iy]
            kappa_val = kappa_map[ix, iy]

            ds = step_size
            d_tau = kappa_val * ds
            dI = j_val * jnp.exp(-tau) * ds

            I_new = I + dI
            tau_new = tau + d_tau

            # Update radiation field
            J = J.at[ix, iy].add(dI)

            x_new = x + direction[0] * ds
            y_new = y + direction[1] * ds

            return (x_new, y_new, I_new, tau_new, J)

        # Initial state
        x0, y0 = source_pos
        initial = (x0, y0, 0.0, 0.0, J)
        _, _, _, _, J_new = jax.lax.fori_loop(0, max_steps, body_fn, initial)
        return J_new

    # Run over all directions
    for i in range(num_rays):
        J = trace_and_accumulate(J, directions[i])

    return J


def compute_radiation_field_from_multiple_stars(
    j_map, kappa_map, source_positions,
    num_rays=360, step_size=0.5, max_steps=1000
):
    """
    Compute radiation field J(x, y) from multiple point sources.

    Args:
        j_map: 2D emissivity field
        kappa_map: 2D absorption field
        source_positions: list or array of (x, y) star positions
        num_rays: rays per star
        step_size: ray marching step
        max_steps: steps per ray

    Returns:
        J: total radiation field from all sources
    """
    J_total = jnp.zeros_like(j_map)
    mean_J  = 0
    for source_pos in source_positions:
        J_single = compute_radiation_field_from_star(
            j_map, kappa_map, source_pos,
            num_rays=num_rays,
            step_size=step_size,
            max_steps=max_steps
        )
        J_total += J_single
        mean_J  += jnp.mean(J_single)
        #print('mean J:', mean_J)
    return J_total



def gaussian_emissivity(Nx, Ny, center, amplitude=1e3, width=5.0):
    """
    Generate a 2D Gaussian emissivity profile centered at a given position.

    Args:
        Nx, Ny: grid size
        center: (x0, y0) position of the source
        amplitude: peak intensity
        width: standard deviation (in pixels)

    Returns:
        emissivity: 2D array (Nx, Ny)
    """
    X, Y = jnp.meshgrid(jnp.arange(Nx), jnp.arange(Ny), indexing='ij')
    x0, y0 = center
    r2 = (X - x0)**2 + (Y - y0)**2
    return amplitude * jnp.exp(-r2 / (2 * width**2))



if __name__ == "__main__":

    

    # Generate an emissivity fields
    key = random.PRNGKey(42)

    Nx, Ny = 200, 200



    kappa, mask = generate_correlated_lognormal_field(key, shape=(Nx, Ny), mean=1.0, length_scale=0.05, sigma_g=1.2)


    # Extract coordinates of top 1% (True values in mask)
    star_indices = jnp.argwhere(mask)

    star_positions = star_indices.astype(jnp.float32) + 0.5  # (x, y) coords


    emissivity = jnp.zeros((Nx, Ny))

    # could vmapp emissivity and than take sum over output
    for pos in star_positions:
        emissivity += gaussian_emissivity(Nx, Ny, center=pos, amplitude=1e3, width=5.0)

    

    J_multi = compute_radiation_field_from_multiple_stars(
        emissivity, kappa, star_positions,
        num_rays=360, step_size=0.2, max_steps=1500
    )

    # Plot the result as an image
    plt.figure(figsize=(6, 5))
    plt.imshow(jnp.log10(J_multi + 1e-6), origin='lower', cmap='inferno')
    plt.title("Ray-Traced Intensity Image")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="log10(Intensity)")
    plt.tight_layout()
    plt.show()
    plt.savefig('ray-tracing.png')

