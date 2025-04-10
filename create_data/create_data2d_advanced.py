import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Set the GPU device to use (0, 1, etc.)
import matplotlib.pyplot as plt
from create_data.create_turbulent_2D import generate_correlated_lognormal_field
from ray_tracing_many_stars import compute_radiation_field_from_multiple_stars, gaussian_emissivity
import time
import jax
import jax.numpy as jnp
from jax import random, vmap



compute_radiation_field_from_multiple_stars_vmapped = vmap(compute_radiation_field_from_multiple_stars, in_axes=(0, 0, 0, None, None, None))
generate_correlated_lognormal_field_vmapped = vmap(generate_correlated_lognormal_field, in_axes=(0, None, None, None, None))
gaussian_emissivity_vmapped = vmap(gaussian_emissivity, in_axes=(None, None, 0, None, None))
gaussian_emissivity_vmapped_vmapped = vmap(gaussian_emissivity_vmapped, in_axes=(None, None, 0, None, None))

# also jit them


def create_data_2d_advanced(keys, Nx=200, Ny=200, num_rays=360,step_size=0.2, max_steps=1500):

    #Nx, Ny = 200, 200

    print("keys shape:", keys.shape)
    kappas, masks = generate_correlated_lognormal_field_vmapped(keys, (Nx, Ny), 1.0, 0.05, 1.2)
    #kappa, mask = generate_correlated_lognormal_field(key, shape=(Nx, Ny), mean=1.0, length_scale=0.05, sigma_g=1.2)

    print('kappas:', kappas.shape)
    print('masks:', masks.shape)    


    # Extract coordinates of top 1% (True values in mask)
    star_indices = jnp.array([jnp.argwhere(mask) for mask in masks])  # this probably needs to be written more efficiently (masks all shape len 400 because 400 brightest pixels are taken)

    #star_indices = jnp.argwhere(mask)

    print('star_indices:', star_indices.shape)

    star_positions = star_indices.astype(jnp.float32) + 0.5  # (x, y) coords

    print('star_positions:', star_positions.shape)

    emissivities = gaussian_emissivity_vmapped_vmapped(Nx, Ny, star_positions, 1e3, 5.0)
    emissivities = jnp.sum(emissivities, axis=1)

    print('emissivity:', emissivities.shape)
    """
    #emissivity = jnp.zeros((Nx, Ny))

    #for pos in star_positions:
    #    emissivity += gaussian_emissivity(Nx, Ny, center=pos, amplitude=1e3, width=5.0)
    """
    J_multis = compute_radiation_field_from_multiple_stars_vmapped(
        emissivities, kappas, star_positions,
        num_rays, step_size, max_steps
    )

    return jnp.array([J_multis, kappas, masks, star_positions, emissivities])

""" 
#create_data_2d_advanced_vmapped = vmap(create_data_2d_advanced, in_axes=(0, None, None, None, None, None))

#create_data_2d_advanced_vmapped_jitted = jax.jit(create_data_2d_advanced_vmapped)


"""
def create_data_2d_advanced_different_try(key, Nx=200, Ny=200, num_rays=360,step_size=0.2, max_steps=1500):  #  remove '='




    kappa, mask = generate_correlated_lognormal_field(key, shape=(Nx, Ny), mean=1.0, length_scale=0.05, sigma_g=1.2)


    # Extract coordinates of top 1% (True values in mask)
    def fixed_argwhere(mask):  # change 400 to non hardcoded-value 
        coords = jnp.argwhere(mask, size=400)
        return coords 
    star_indices = fixed_argwhere(mask) 

    star_positions = star_indices.astype(jnp.float32) + 0.5  # (x, y) coords

    emissivity = jnp.zeros((Nx, Ny))
    # could vmap emissivity and than take sum over output
    #emissivity = gaussian_emissivity_vmapped(Nx, Ny, star_positions, 1e3, 5.0)
    #emissivity = jnp.sum(emissivity, axis=0)

    def pos_loop(state, pos):
        emissivity = state 
        return emissivity + gaussian_emissivity(Nx, Ny, center=pos, amplitude=1e3, width=5.0), None

    emissivity, _ = jax.lax.scan(pos_loop, emissivity, star_positions)
    #for pos in star_positions:
    #    emissivity += gaussian_emissivity(Nx, Ny, center=pos, amplitude=1e3, width=5.0)

    tstart = time.time()
    J_multi = compute_radiation_field_from_multiple_stars(
        emissivity, kappa, star_positions,
        360, 0.2, 1500
    )  #add parameters of function instead of numbers 
    tend = time.time()
    print("total time:", tend - tstart)
    return J_multi, kappa, mask, star_positions, emissivity

create_data_2d_advanced_different_try_vmapped = vmap(create_data_2d_advanced_different_try, in_axes=(0, None, None, None, None, None))



if __name__ == "__main__":

    
    #keys = random.split(random.PRNGKey(42), 10)
    print("start of main")
    print("jax devices:", jax.devices())
    
    key = random.PRNGKey(42)
    keys = random.split(random.PRNGKey(42), 1)
    #testarray = jnp.ones((2,2))
    #gaussian_emissivity_vmapped(10, 10, testarray, 1e3, 5.0)
    results = jax.jit(create_data_2d_advanced_different_try_vmapped(keys, 200, 200, 360, 0.2, 1500), device=jax.devices()[5])
    #data = create_data_2d_advanced(keys, 200, 200, 360, 0.2, 1500)
    #data = create_data_2d_advanced(keys, 200, 200, 360, 0.2, 1500) 
    #jnp.save("advanced_training_data.npy", data)
