import numpy as np
import matplotlib.pyplot as plt

# Parameters
r = 1.5
lam = 1.0
tau = 1.0
gamma = 1.0

# Grid
u = np.linspace(-3, 3, 400)
v = np.linspace(-3, 3, 400)
U, V = np.meshgrid(u, v)

# Probability density function proportional to exp(-E)
P = np.exp(-(0.5 * lam * (r - U * V)**2 + 0.5 * tau * U**2 + 0.5 * gamma * V**2))

fig, ax = plt.subplots(figsize=(6, 5))
contour = ax.contourf(U, V, P, levels=20, cmap='viridis')
ax.contour(U, V, P, levels=10, colors='white', alpha=0.6, linewidths=0.8)
fig.colorbar(contour, ax=ax, label='Density')
ax.set_xlabel('$u$')
ax.set_ylabel('$v$')
ax.set_title(f'Contours of the Latent Space $(u, v)$ for $r = {r}$')
ax.grid(True, linestyle='--', alpha=0.3)
plt.savefig('contours.png', bbox_inches='tight', dpi=150)
plt.close()
print("Plot generated successfully.")