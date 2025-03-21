# RadiativeTransfer

With radiative transfer we describe the transfer of energy in the form of electromagnetic radiation. When radiation travels through a medium it is affected by absorption, emission, and scattering processes. This is described by the following equation:

\[
\frac{1}{c} \frac{\partial I_\nu(\mathbf{n}, \mathbf{x}, t)}{\partial t} + \mathbf{n} \cdot \nabla I_\nu(\mathbf{n}, \mathbf{x}, t) = -\alpha_\nu(\mathbf{x}, t) I_\nu(\mathbf{n}, \mathbf{x}, t) + j_\nu(\mathbf{x}, t) + \text{scattering terms}.
\]

In many cases we can assume that the speed of the propagation of radiation is so large, that photons pass through our object of interest in a time much shorter than that the object can change its properties. This means we can ignore the light travel time effects and regard the radiation as a steady-state flow of photons. The equation for the radiative transfer turns into:

\[
\mathbf{n} \cdot \nabla I_\nu(\mathbf{n}, \mathbf{x}) = -\alpha_\nu(\mathbf{x}) I_\nu(\mathbf{n}, \mathbf{x}) + j_\nu(\mathbf{x}) + \text{scattering terms}.
\]

Expressing this in speherical coordinates we get:

\[
\mu \frac{\partial I_\nu}{\partial r} + \frac{1 - \mu^2}{r} \frac{\partial I_\nu}{\partial \mu} + \frac{1}{r \sin\theta} \frac{\partial I_\nu}{\partial \phi} = -\alpha_\nu I_\nu + j_\nu + \text{scattering terms}.
\]

In spherical coordinates, assuming spherical symmetry, this leads to:

\[
\frac{1}{c}\frac{\partial I_\nu}{\partial t} + \frac{1}{r^2}\frac{\partial}{\partial r}\Bigl(r^2 I_\nu\Bigr) = j_\nu - \alpha_\nu I_\nu.
\]

Here:

 $I_\nu$ is the specific intensity at frequency $nu$, 

 $j_\nu$ is the emission coefficient,

 $alpha_\nu$ is the absorption coefficient,

c is the speed of light,

r is the radial coordinate.


## Sources: 

### Radiative Transfer (RT)
[Lecture "Radiative transfer in astrophysics" by C.P. Dullemond](https://www.ita.uni-heidelberg.de/~dullemond/lectures/radtrans_2017/index.shtml?lang=en)

### Neural operators:
[Operator Learning: Algorithms and Analysis](https://arxiv.org/abs/2402.15715)

### Code examples:
[neuraloperator](https://github.com/neuraloperator/neuraloperator)

[simple_FNO_in_JAX](https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/neural_operators/simple_FNO_in_JAX.ipynb)






This sentence uses `$` delimiters to show math inline: $\sqrt{3x-1}+(1+x)^2$


Form in Kugelkoordinaten angeben


