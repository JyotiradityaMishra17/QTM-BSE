import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ———— global font setup ————
mpl.rcParams['font.family']       = 'serif'
mpl.rcParams['font.serif']        = ['CMU Serif']
mpl.rcParams['axes.unicode_minus'] = False    # fallback to ASCII hyphen

# load your data
data    = np.loadtxt("absorption_eh.dat")
omega   = data[:, 0]
eps2    = data[:, 1]

data_2  = np.loadtxt("absorption_noeh.dat")
omega_2 = data_2[:, 0]
eps2_2  = data_2[:, 1]

# create the figure
plt.figure(figsize=(8, 6))

plt.plot(omega,   eps2,   linewidth=1, label='Interacting', color = 'red')
plt.plot(omega_2, eps2_2, linewidth=1, linestyle='--', label='Non-interacting', color = 'black')

# math‐mode labels
plt.xlabel(r'$\omega$',    fontsize=14)
plt.ylabel(r'$\epsilon_{2}(\omega)$', fontsize=14)

# title in normal text but CMU Serif
plt.title('Absorption Spectrum', fontsize=16)

plt.grid(True)
plt.legend(fontsize=12)
plt.xlim(0, 25)

# save high-res PNG
plt.savefig('absorption_eh.png', dpi=300, bbox_inches='tight')
plt.show()

# Find the maximum values of eps2 and eps2_2
max_eps2 = np.max(eps2)
max_eps2_2 = np.max(eps2_2)

# Print the maximum values
print("Maximum value of eps2 (Interacting):", max_eps2)
print("Maximum value of eps2_2 (Non-interacting):", max_eps2_2)




# data_eig_noeh = np.loadtxt("eigenvalues_noeh.dat", comments="#")
# eigenvalues_noeh = data_eig_noeh[:, 6]
# sorted_eigenvalues_noeh = np.sort(eigenvalues_noeh)

# # Print the sorted eigenvalues
# print("Sorted eigenvalues (no e-h interaction) (in eV):")
# print(sorted_eigenvalues_noeh[:20])

# data_eig = np.loadtxt("eigenvalues.dat", comments="#")
# eigenvalues = data_eig[:, 0]
# sorted_eigenvalues = np.sort(eigenvalues)
# # Print the sorted eigenvalues
# print("Sorted eigenvalues (with e-h interaction) (in eV):")
# print(sorted_eigenvalues[:20])

# # Save the eigenvalues in a .dat file
# np.savetxt("sorted_eigenvalues_noeh.dat", sorted_eigenvalues_noeh, header="Sorted eigenvalues (no e-h interaction) (in eV)", comments="#")
# np.savetxt("sorted_eigenvalues.dat", sorted_eigenvalues, header="Sorted eigenvalues (with e-h interaction) (in eV)", comments="#")





