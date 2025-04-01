# Radiative Transfer

With radiative transfer we describe the transfer of energy in the form of electromagnetic radiation. When radiation travels through a medium it is affected by absorption, emission, and scattering processes. This is described by the following equation:


$$\frac{1}{c} \frac{\partial I_\nu(\mathbf{n}, \mathbf{x}, t)}{\partial t} + \mathbf{n} \cdot \nabla I_\nu(\mathbf{n}, \mathbf{x}, t) = -\alpha_\nu(\mathbf{x}, t) I_\nu(\mathbf{n}, \mathbf{x}, t) + j_\nu(\mathbf{x}, t) + \text{scattering terms}$$


## Radiative Transfer in the static case, assuming spherical symmetry
In many cases we can assume that the speed of the propagation of radiation is so large, that photons pass through our object of interest in a time much shorter than that the object can change its properties. This means we can ignore the light travel time effects and regard the radiation as a steady-state flow of photons. The equation for the radiative transfer turns into:


$$\mathbf{n} \cdot \nabla I_\nu(\mathbf{n}, \mathbf{x}) = -\alpha_\nu(\mathbf{x}) I_\nu(\mathbf{n}, \mathbf{x}) + j_\nu(\mathbf{x}) + \text{scattering terms}$$


In spherical coordinates, assuming spherical symmetry (and neglecting scattering terms), this leads to:

$$\frac{\partial I_\nu}{\partial r} + \frac{1}{r^2}\frac{\partial}{\partial r}\Bigl(r^2 I_\nu\Bigr) = j_\nu - \alpha_\nu I_\nu$$

Here:

 $I_\nu$ is the specific intensity at frequency $\nu$, 

 $j_\nu$ is the emission coefficient,

 $\alpha_\nu$ is the absorption coefficient,

c is the speed of light,

r is the radial coordinate.


This is an ODE that can be solved analytically (if $\alpha_\nu$ and  $j_\nu$ are constant) or numercially. In [RTE_spherical_symmetry](https://github.com/RuneRost/RadiativeTransfer/blob/main/RTE_sphercial_symmetry.ipynb) I implemented an analytic and a numeric solution which can be used to generate training data. (ADD DERIVATION FOR ANALYTIC SOLUTION)



## Operator Learning

Neural operators are a specific class of deep learning architectures  ([Wikipedia](https://en.wikipedia.org/wiki/Neural_operators)). In general operator learning aims to discover properties of an underlying dynamical system or partial differential equation (PDE) from data. Instead of learning a mapping between discrete vector spaces they learn mappings between (infinite-dimensional) function spaces. They are build by a composition of integral operators and nonlinear functions,  which results in the following recursive definition at layer i:

$$u_{i+1}(x) = \sigma \left( \int_{\Omega_i} K^{(i)}(x,y)\, u_i(y)\, dy + b_i(x) \right), \quad x \in \Omega_{i+1} $$

where $\Omega_{i} \in \mathbb{R}^{d_i}$ di is a compact domain, $b_{i}$ is a bias function, and $K(i)$ is the kernel. The kernels and biases are then parameterized and trained similarly to standard neural networks. However, approximating the kernels or evaluating the integral operators could be computationally expensive. ([A Mathematical Guide to Operator Learning](https://arxiv.org/pdf/2312.14688)). Hence, several neural operator architectures have been proposed to overcome these challenges, such as DeepONets (Lu et al., 2021a) and Fourier neural operators (Li et al., 2021a). 

### Architectures

#### FNOs

#### PCA-Net

#### DeepO-Net

---------------------------


Beschreibung der einzelnen `Dateien`, die ich in diesem Framework habe noch hinzufügen (mit Verlinkung)

### Sources: 

#### Radiative Transfer (RT)
[Lecture "Radiative transfer in astrophysics" by C.P. Dullemond](https://www.ita.uni-heidelberg.de/~dullemond/lectures/radtrans_2017/index.shtml?lang=en)

#### Neural operators:
[Operator Learning: Algorithms and Analysis](https://arxiv.org/abs/2402.15715)

[A Mathematical Guide to Operator Learning](https://arxiv.org/html/2312.14688v1)

#### Code examples:
[neuraloperator](https://github.com/neuraloperator/neuraloperator)

[simple_FNO_in_JAX](https://github.com/Ceyron/machine-learning-and-simulation/blob/main/english/neural_operators/simple_FNO_in_JAX.ipynb)



`$` `$` `$`





