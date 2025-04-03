import numpy as np
import matplotlib.pyplot as plt

# Grid and parameters
Nx, Ny = 100, 100
Lx, Ly = 1.0, 1.0
dx, dy = Lx / Nx, Ly / Ny
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')

# Opacity and emissivity
kappa = 1.0 ## here your density field
j0 = 10.0
xc, yc = Lx / 2, Ly / 2
sigma = 0.05
j_emissivity = j0 * np.exp(-((X - xc)**2 + (Y - yc)**2) / (2 * sigma**2))

# Angular discretization
N_theta = 16
theta_list = np.linspace(0, 2 * np.pi, N_theta, endpoint=False)

# Storage for total intensity
J = np.zeros((Nx, Ny))

# Solve RTE for each angle
for theta in theta_list:
    mu_x = np.cos(theta)
    mu_y = np.sin(theta)
    I = np.zeros((Nx, Ny))
    
    # Determine sweep order based on angle
    i_range = range(Nx) if mu_x >= 0 else range(Nx - 1, -1, -1)
    j_range = range(Ny) if mu_y >= 0 else range(Ny - 1, -1, -1)
    
    for i in i_range:
        for j in j_range:
            if (i - np.sign(mu_x) < 0 or i - np.sign(mu_x) >= Nx or
                j - np.sign(mu_y) < 0 or j - np.sign(mu_y) >= Ny):
                I_up_x = 0.0
                I_up_y = 0.0
            else:
                I_up_x = I[int(i - np.sign(mu_x)), j]
                I_up_y = I[i, int(j - np.sign(mu_y))]
            denom = abs(mu_x) / dx + abs(mu_y) / dy + kappa
            I_avg = (abs(mu_x) * I_up_x / dx + abs(mu_y) * I_up_y / dy) / denom
            source = j_emissivity[i, j] / denom
            I[i, j] = I_avg + source
    
    J += I  # Accumulate for mean intensity

# Compute mean intensity
J /= N_theta

# Plot
plt.figure(figsize=(6, 5))
plt.imshow(J.T, origin='lower', extent=[0, Lx, 0, Ly], aspect='auto', cmap='inferno')
plt.title(f"Mean Intensity J(x, y)\nwith {N_theta} Angles and Central Emissivity")
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(label="Mean Intensity")
plt.tight_layout()
plt.show()

