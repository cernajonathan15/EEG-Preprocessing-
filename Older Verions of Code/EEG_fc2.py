# Functional Connectivity at the Level of Networks

# Importing necessary libraries and functions
import requests
import os
import os.path as op
from mne.datasets import fetch_fsaverage

# Define a function to download files from GitHub
def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(destination, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=128):
            fd.write(chunk)

# Assuming you've already set the 'fsaverage' folder:
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
labels_dir = os.path.join(subjects_dir, 'fsaverage', 'label', 'Brainnetome')  # Adjusted the path to include 'Brainnetome'
os.makedirs(labels_dir, exist_ok=True)

# Update the base_url to point to the Brainnetome atlas repository
base_url = "https://raw.githubusercontent.com/brainnetome/bnatlasviewer/222a037a2295421156b50f4ec269ecdab59ebef0/content"

# List of Brainnetome atlas files you want to download
files = [
    "bnatlas.nii.gz",
    "bnatlas.nii.txt",
    "bnatlas_tree.csv",
    "bnatlas_tree.json",
    # Add any other files you want to download from the repository
]

for file in files:
    file_url = f"{base_url}/{file}"
    dest_path = os.path.join(labels_dir, file)
    download_file(file_url, dest_path)

# Check if files have been downloaded successfully
all_files_exist = all([os.path.exists(op.join(labels_dir, file)) for file in files])

if all_files_exist:
    print("Files downloaded successfully!")
else:
    print("Some files are missing!")

#################################################################################
# Set needed directories, load inverse solution epoched data

# Import necessary libraries and functions
import numpy as np
import mne
from gc import collect

# Setting directories and file names
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # --> replace participant ID here

# Define frequency bands at the top level
fmin = [1, 4, 8, 13, 30]
fmax = [4, 8, 13, 30, 50]
band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# Load a subset of the computed inverse solution data across epochs using a generator
def load_data_in_batches(batch_size):
    epoch_num = 0

    while True:
        stcs = []  # List to store source estimates for each batch
        for _ in range(batch_size):
            inverse_solution_file = os.path.join(output_dir, f"{subj}_inversesolution_epoch{epoch_num}.fif-stc.h5")

            # Check if the file exists
            if not os.path.exists(inverse_solution_file):
                break

            stc_epoch = mne.read_source_estimate(inverse_solution_file)
            stcs.append(stc_epoch)
            epoch_num += 1

        # If no epochs were loaded in this iteration, yield None and break
        if not stcs:
            yield None
            break

        yield stcs

# Initialize variables to compute overall mean and standard deviation
num_batches = 0
overall_mean_sum = 0
overall_std_sum = 0

for stcs in load_data_in_batches(10):
    if stcs is None:
        break

    # Convert the list of VectorSourceEstimate objects in the current batch to a 3D numpy array
    data_array = np.array([stc.data for stc in stcs])

    # Update the overall mean and standard deviation
    overall_mean_sum += np.mean(data_array, axis=0)
    overall_std_sum += np.std(data_array, axis=0)
    num_batches += 1

    # Manually invoke garbage collection
    collect()

# Compute the overall mean and standard deviation across all batches
overall_mean = overall_mean_sum / num_batches
overall_std = overall_std_sum / num_batches

# Compute the relative variability (coefficient of variation) using the overall mean and standard deviation
coeff_var = overall_std / overall_mean

# Compute the average coefficient of variation across all source points
avg_coeff_var = np.mean(coeff_var)

print(f"Average Coefficient of Variation: {avg_coeff_var:.2f}")
#################################################################################

# Compute Graph Theory Metrics

import sys
sys.path.append("C:\\Users\\cerna\\anaconda3\\envs\\pythonProject1\\Lib\\site-packages")
import networkx as nx
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import hilbert

# Define a function to compute dPLI matrices for each frequency band
def compute_dPLI(data):
    print("Starting dPLI computation...")
    # Use the globally defined frequency bands
    dpli_bands = {}
    for i, band in enumerate(band_names):
        print(f"Processing frequency band: {band}")

        # Define the sampling frequency
        sample_rate = 512

        # Filter the data for the current frequency band
        data_band = np.empty_like(data)
        for ch in range(data.shape[1]):
            data_band[:, ch, :] = mne.filter.filter_data(data[:, ch, :], sfreq=sample_rate,
                                                         l_freq=fmin[i], h_freq=fmax[i], method='iir', verbose=False)

        # Compute the instantaneous phase using Hilbert transform
        phase_data = np.angle(hilbert(data_band, axis=-1))

        # Initialize dPLI matrix for the current frequency band
        dpli_matrix = np.zeros((data.shape[1], data.shape[1]))

        for ch1 in range(data.shape[1]):
            for ch2 in range(data.shape[1]):
                # Compute the phase difference
                phase_diff = phase_data[:, ch1, :] - phase_data[:, ch2, :]

                # Compute dPLI
                dpli_matrix[ch1, ch2] = np.abs(np.mean(np.sign(phase_diff)))

        dpli_bands[band] = dpli_matrix

    print("Finished dPLI computation for current batch.")
    return dpli_bands


# Compute dPLI matrices for each batch
dpli_matrices_bands = []
batch_count = 0
for batch in load_data_in_batches(10):
    if batch is None:
        break
    batch_count += 1
    print(f"Processing batch {batch_count}...")
    dpli_matrices_bands.append(compute_dPLI(np.array([stc.data for stc in batch])))

# Initialize dictionaries to store graph metrics for each frequency band
L_values = {}
C_values = {}
global_efficiency_values = {}
sigma_values = {}

for band in band_names:
    # Extract the dPLI matrices for the current frequency band from each batch
    dpli_matrices = [dpli_band[band] for dpli_band in dpli_matrices_bands]

    # Average the dPLI matrices across batches for the current frequency band
    avg_dpli_matrix = np.mean(dpli_matrices, axis=0)

    # Create a directed graph from the averaged dPLI matrix
    G = nx.from_numpy_matrix(avg_dpli_matrix, create_using=nx.DiGraph)

    # Compute graph theory metrics for the directed graph

    # 1. Characteristic Path Length
    L_values[band] = nx.average_shortest_path_length(G)

    # 2. Directed Clustering Coefficient
    C_values[band] = nx.average_clustering(G)

    # 3. Global Efficiency
    global_efficiency_values[band] = nx.global_efficiency(G)

    # Compute small-worldness
    # Generate random directed graphs and compute C_random and L_random
    num_random_graphs = 10  # Adjust based on computational feasibility
    C_random_vals = []
    L_random_vals = []

    for _ in range(num_random_graphs):
        G_random = nx.directed_configuration_model(list(G.in_degree()), list(G.out_degree()), create_using=nx.DiGraph)
        G_random = nx.DiGraph(G_random)  # Remove parallel edges
        G_random.remove_edges_from(nx.selfloop_edges(G_random))  # Remove self-loops

        C_random_vals.append(nx.average_clustering(G_random))
        try:
            L_random_vals.append(nx.average_shortest_path_length(G_random))
        except nx.NetworkXError:  # Catch error if the graph is not strongly connected
            pass

    C_random = np.mean(C_random_vals)
    L_random = np.mean(L_random_vals)

    # Compute small-worldness for the current frequency band
    sigma_values[band] = (C_values[band] / C_random) / (L_values[band] / L_random)

# Print the computed graph metrics for each frequency band
for band in band_names:
    print(f"Frequency Band: {band}")
    print(f"Characteristic Path Length: {L_values[band]}")
    print(f"Global Efficiency: {global_efficiency_values[band]}")
    print(f"Average Directed Clustering Coefficient: {C_values[band]}")
    print(f"Small-Worldness: {sigma_values[band]}")
    print("-" * 50)

#################################################################################
