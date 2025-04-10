import numpy as np

def load_v_coul(filename):
    """
    Load Coulomb potentials from a file and arrange them as a 2D array.
    
    The file should have 7 columns with the following format:
      - Columns 0, 1, 2: components of the q vector.
      - Columns 3, 4, 5: components of the g vector.
      - Column 6: Coulomb potential value.
      
    The returned array has shape (n_q, n_g) where n_q is the number of unique q vectors,
    and each row contains the Coulomb potentials for the corresponding q vector.
    
    Parameters
    ----------
    filename : str
        The path to the input file.
        
    Returns
    -------
    v_coul_array : numpy.ndarray
        A 2D array with the Coulomb potentials arranged with q-index as first dimension,
        and g-index as the second.
        
    Raises
    ------
    ValueError
        If the number of g-index entries is not consistent across different q vectors.
    """
    # Load the data from the file.
    data = np.loadtxt(filename)
    
    # Ensure data is 2D even if file contains only one row.
    if data.ndim == 1:
        data = data.reshape((1, -1))
    
    # Extract q vectors, g vectors, and Coulomb potential values.
    q_vectors = data[:, :3]
    potentials = data[:, 6]
    
    # Identify unique q vectors. `q_indices` assigns each row a q vector index.
    unique_q, q_indices = np.unique(q_vectors, axis=0, return_inverse=True)
    num_unique_q = unique_q.shape[0]
    
    # Group the Coulomb potentials by unique q vector.
    potentials_by_q = []
    for qi in range(num_unique_q):
        indices = np.where(q_indices == qi)[0]
        potentials_by_q.append(potentials[indices])
    
    # Check that all q vectors have the same number of g entries.
    n_entries = len(potentials_by_q[0])
    if not all(len(arr) == n_entries for arr in potentials_by_q):
        raise ValueError("Inconsistent number of g-index entries for different q vectors.")
    
    # Convert list to a 2D numpy array.
    v_coul_array = np.array(potentials_by_q)
    
    return v_coul_array

# # Example usage:
# if __name__ == '__main__':
#     filename = "v_coul"  # Replace with your actual file path if needed.
#     v_coul_arr = load_v_coul(filename)
#     print("Coulomb potential array (shape {}):".format(v_coul_arr.shape))
#     print(v_coul_arr)
