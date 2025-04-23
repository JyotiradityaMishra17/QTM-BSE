import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("absorption_eh.dat")
omega = data[:, 0]
eps2 = data[:, 1]

data_2 = np.loadtxt("absorption_noeh.dat")
omega_2 = data_2[:, 0]
eps2_2 = data_2[:, 1]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(omega, eps2,linewidth = 2, label='EH')
plt.plot(omega_2, eps2_2, linewidth = 2, linestyle='--', label='EH=0')
plt.xlabel('omega')
plt.ylabel('eps2')
plt.title('Plot of omega vs eps2')
plt.grid(True)
plt.legend()
plt.savefig('absorption_eh.png', dpi=300, bbox_inches='tight')



data_eig_noeh = np.loadtxt("eigenvalues_noeh.dat", comments="#")
eigenvalues_noeh = data_eig_noeh[:, 6]
sorted_eigenvalues_noeh = np.sort(eigenvalues_noeh)

# Print the sorted eigenvalues
print("Sorted eigenvalues (no e-h interaction) (in eV):")
print(sorted_eigenvalues_noeh[:20])

data_eig = np.loadtxt("eigenvalues.dat", comments="#")
eigenvalues = data_eig[:, 0]
sorted_eigenvalues = np.sort(eigenvalues)
# Print the sorted eigenvalues
print("Sorted eigenvalues (with e-h interaction) (in eV):")
print(sorted_eigenvalues[:20])

# Save the eigenvalues in a .dat file
np.savetxt("sorted_eigenvalues_noeh.dat", sorted_eigenvalues_noeh, header="Sorted eigenvalues (no e-h interaction) (in eV)", comments="#")
np.savetxt("sorted_eigenvalues.dat", sorted_eigenvalues, header="Sorted eigenvalues (with e-h interaction) (in eV)", comments="#")





