"""
npy_to_txt.py: Convert higher-dimensional .npy files to txt files.

The primary motivation of this script is for easy importing into C++ projects.
The output format is as follows:
- The first line is the number of dimensions.
- The next lines each contain a dimension size.
- Subsequent lines contain the flattened elements of the array, with each line
  representing the innermost dimension's elements.

Example:
For a 4D array of shape (2, 3, 4, 5), the .txt file starts with:
4
2
3
4
5
Followed by lines each containing 5 space-separated float values.
"""

import numpy as np
import sys

def save_arr(txt_file, data):
    """Writes the dimensional header of the array to the specified text file.
    
    Args:
        txt_file (str): Path to the output text file.
        data (np.ndarray): The numpy array to extract dimensions from.
    """
    with open(txt_file, 'w') as f:
        shape = data.shape
        # Write the number of dimensions
        f.write(f"{len(shape)}\n")
        # Write each dimension value on a new line
        for dim in shape:
            f.write(f"{dim}\n")

def npy_to_txt(npy_file, txt_file):
    """Converts a .npy file to a .txt file with the specified format.
    
    Args:
        npy_file (str): Path to the input .npy file.
        txt_file (str): Path to the output .txt file.
    """
    # Load the .npy file
    data = np.load(npy_file)
    
    # Save the dimensional header
    save_arr(txt_file, data)
    
    # Reshape the data to 2D, preserving the innermost dimension
    reshaped_data = data.reshape(-1, data.shape[-1])
    
    # Append the reshaped data to the text file
    with open(txt_file, 'a') as f:
        np.savetxt(f, reshaped_data, fmt='%f', delimiter=' ')

def main():
    """Handles command-line arguments and invokes the conversion."""
    # Default file paths
    npy_file = '../assets/expected_preprocessed_image.npy'
    txt_file = npy_file[:-4] + '.txt'
    
    # Parse command-line arguments
    if len(sys.argv) == 1:
        # Use default paths if no arguments are provided
        pass
    elif len(sys.argv) == 3:
        npy_file = sys.argv[1]
        txt_file = sys.argv[2]
    else:
        print("Usage: python npy_to_txt.py [<npy_path> <txt_path>]")
        return
    
    # Perform the conversion
    npy_to_txt(npy_file, txt_file)
    print(f"Converted {npy_file} to {txt_file}")

if __name__ == '__main__':
    main()