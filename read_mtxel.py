import re
import numpy as np

def parse_line(line):
    """
    Parse a single line of the form:
      matrixName( index1, index2, index3, index4 ) = (real_value,imag_value)
    Returns a tuple: (matrix_name, (index1,index2,index3,index4), complex_value)
    or None if the line does not match.
    """
    pattern = r'(\w+)\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*\(([^,]+),([^)]*)\)'
    match = re.search(pattern, line)
    if match:
        matrix_name = match.group(1)
        # indices are extracted as strings; convert to integers.
        indices = tuple(int(match.group(i)) for i in range(2, 6))
        # Convert the real and imaginary parts to float and then construct a complex number.
        real_part = float(match.group(6))
        imag_part = float(match.group(7))
        return matrix_name, indices, complex(real_part, imag_part)
    return None

def parse_file(filename):
    """
    Read the file and extract all matrix entries, storing them in dictionaries.
    Returns a dictionary mapping each matrix name to its complete NumPy array.
    """
    # We'll store the extracted entries for each matrix here.
    matrix_entries = {}
    # Also keep track of the maximum index in each of the four dimensions for each matrix.
    max_indices = {}

    with open(filename, 'r') as f:
        for line in f:
            parsed = parse_line(line)
            if parsed:
                matrix_name, indices, cvalue = parsed
                # Initialize lists/dictionaries if needed.
                if matrix_name not in matrix_entries:
                    matrix_entries[matrix_name] = []
                    max_indices[matrix_name] = [0, 0, 0, 0]
                matrix_entries[matrix_name].append((indices, cvalue))
                # Update maximum indices; our indices are 1-indexed.
                for i in range(4):
                    if indices[i] > max_indices[matrix_name][i]:
                        max_indices[matrix_name][i] = indices[i]

    # Allocate NumPy arrays with the determined shapes and fill in the values.
    arrays = {}
    for name, entries in matrix_entries.items():
        # The shape is determined by the max index along each dimension.
        shape = tuple(max_indices[name])
        # Create an array of the proper shape, using complex128 for double-precision complex values.
        arr = np.zeros(shape, dtype=np.complex128)
        for indices, cvalue in entries:
            # Convert from 1-indexing (in the file) to 0-indexing (in Python).
            idx = tuple(i-1 for i in indices)
            arr[idx] = cvalue
        arrays[name] = arr

    return arrays
