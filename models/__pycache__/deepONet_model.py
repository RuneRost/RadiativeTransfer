# code for the DeepONet is from here: https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/neural_operators/simple_deepOnet_in_JAX.ipynb
# -> use other models presented there for further testing

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx



class DeepONet1d(eqx.Module):
    branch_net: eqx.nn.MLP
    trunk_net: eqx.nn.MLP
    bias: jax.Array

    def __init__(
        self,
        in_size_branch,
        width_size,
        depth,
        interact_size,
        activation,
        *,
        key,
    ):
        """
        For simplicity, branch and trunk MLP are configured similarly
        """
        b_key, t_key = jr.split(key)
        self.branch_net = eqx.nn.MLP(
            in_size=in_size_branch,
            out_size=interact_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=b_key,
        )
        self.trunk_net = eqx.nn.MLP(
            in_size=1,
            out_size=interact_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=activation,
            key=t_key,
        )
        self.bias = jnp.zeros((1,))

    def __call__(self, x_branch, x_trunk):
        """
        x_branch.shape = (in_size_branch,)
        x_trunk.shape = (1,)

        return shape: "scalar"
        """
        branch_out = self.branch_net(x_branch)
        trunk_out = self.trunk_net(x_trunk)
        inner_product = jnp.sum(branch_out * trunk_out, keepdims=True)

        return (inner_product + self.bias)[0]