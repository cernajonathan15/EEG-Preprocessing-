# Functional Connectivity (Global)

# Importing necessary libraries and functions
import requests
import os
import os.path as op

# Create a function to download files --> NOTE THAT SOME DOWNLOADS ARE NOT WORKING (e.g., BNA_subregions.xlsx) and need to be downloaded manually
# def download_file(url, destination):
#     response = requests.get(url, stream=True)
#     response.raise_for_status()
#     with open(destination, 'wb') as fd:
#         for chunk in response.iter_content(chunk_size=128):
#             fd.write(chunk)

# Define the base URL for Brainnetome atlas downloads
# base_url = "https://pan.cstcloud.cn/web/share.html?hash="  # Base URL for download links

# Define the destination directory for Brainnetome atlas files
# brainnetome_dir = r'C:\Users\cerna\Downloads'  # Update this with your preferred directory

# Create the destination directory if it doesn't exist
# os.makedirs(brainnetome_dir, exist_ok=True)

# List of Brainnetome atlas files you want to download with desired file names
# files_to_download = [
#     {"hash": "H9wCmuJSFI", "name": "BNA_MPM_thr25_1.25mm.nii.gz"},
#     {"hash": "6eRCJ0zDTFk", "name": "BNA_subregions.xlsx"},
#     {"hash": "JX89RKcgTHM", "name": "BNA_matrix_binary_246x246.csv"},
#     {"hash": "jbLxOO7cRbo", "name": "BNA_PM_4D.nii.gz"},
#     {"hash": "KwqIPh1SoU", "name": "BNA_SC_4D.nii.gz"},
#     {"hash": "GP3urQok", "name": "BNA_FC_4D.nii.gz"},
#     {"hash": "cRrY3Ys6S4g", "name": "HCP40_MNI152_1.25mm.nii.gz"},
# ]

# Download each file
# for file_info in files_to_download:
#     file_hash = file_info["hash"]
#     file_name = file_info["name"]
#     file_url = f"{base_url}{file_hash}"
#     destination_path = os.path.join(brainnetome_dir, file_name)
#     download_file(file_url, destination_path)

# Check if files have been downloaded successfully
# all_files_exist = all([os.path.exists(os.path.join(brainnetome_dir, file_info["name"])) for file_info in files_to_download])

# if all_files_exist:
#     print("Brainnetome atlas files downloaded successfully!")
# else:
#     print("Some files are missing!")

#######################################################################################################################

# DEFINE PARCELS: To define parcels, you need to parse the Brainnetome atlas files
# This file contains information about the parcels, including their labels or indices, coordinates, and potentially other information

# AND

# PARCEL AVERAGING: Once you have defined the parcels, you can proceed with parcel averaging
# Will use the parcels dictionary to determine which parcel each source estimate belongs to and then compute the average for each parcel

# Import necessary libraries
import os
import os.path as op
import mne
import glob
import numpy as np
import pandas as pd
from mne.datasets import fetch_fsaverage

# Setting directories and file names
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # --> replace participant ID here
fs_dir = fetch_fsaverage(verbose=True)
fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")

# Define frequency bands at the top level
fmin = [1, 4, 8, 13, 30]
fmax = [4, 8, 13, 30, 50]
band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# Define the batch size (you can change this)
batch_size = 10

# Load the Brainnetome subregions Excel file into a DataFrame
subregions_file = os.path.join(output_dir, "BNA_subregions.xlsx")
subregions_data = pd.read_excel(subregions_file)

# Create a dictionary to map MNI coordinates to parcel labels
coordinate_to_parcel_mapping = {}

for index, row in subregions_data.iterrows():
    label_l = row['Label ID.L']
    label_r = row['Label ID.R']
    coordinates_l = [int(coord) for coord in row['lh.MNI(X,Y,Z)'].split(',')]
    coordinates_r = [int(coord) for coord in row['rh.MNI(X,Y,Z)'].split(',')]

    # Map left hemisphere coordinates to labels
    coordinate_to_parcel_mapping[tuple(coordinates_l)] = label_l
    # Map right hemisphere coordinates to labels
    coordinate_to_parcel_mapping[tuple(coordinates_r)] = label_r

# Create a list of inverse solution file paths for both left and right hemispheres
inverse_solution_files_lh = glob.glob(os.path.join(output_dir, f"{subj}_inversesolution_epoch*.fif-lh.stc"))
inverse_solution_files_rh = glob.glob(os.path.join(output_dir, f"{subj}_inversesolution_epoch*.fif-rh.stc"))

# Initialize a list to store label time courses
label_time_courses = []

# Define labels based on the Brainnetome atlas for both hemispheres
labels = []

for index, row in subregions_data.iterrows():
    label_l = row['Label ID.L']
    label_r = row['Label ID.R']

    # Create labels for the left and right hemispheres
    label_left = f"Left_{label_l}"
    label_right = f"Right_{label_r}"

    # Add both hemisphere labels to the list
    labels.extend([label_left, label_right])

# Initialize lists to store average_stc for each hemisphere
average_stcs_batch_lh = []
average_stcs_batch_rh = []

# Set the 'subject' parameter for the SourceEstimate objects
subject = 'fsaverage'  # Use 'fsaverage' for population-based template

# Load the source space for the specified subject
src = mne.read_source_spaces(fname, patch_stats=False, verbose=None)

# Process data in batches
for batch_num in range(len(inverse_solution_files_lh) // batch_size):
    batch_files_lh = inverse_solution_files_lh[batch_num * batch_size:(batch_num + 1) * batch_size]
    batch_files_rh = inverse_solution_files_rh[batch_num * batch_size:(batch_num + 1) * batch_size]

    stcs_lh = []
    stcs_rh = []

    for file_path_lh, file_path_rh in zip(batch_files_lh, batch_files_rh):
        try:
            stc_epoch_lh = mne.read_source_estimate(file_path_lh)
            stc_epoch_rh = mne.read_source_estimate(file_path_rh)
            stcs_lh.append(stc_epoch_lh)
            stcs_rh.append(stc_epoch_rh)
        except Exception as e:
            print(f"Error loading files {file_path_lh} or {file_path_rh}: {e}")

    if stcs_lh and stcs_rh:
        # Parcel Averaging
        parcel_data_lh = {}
        parcel_data_rh = {}

        for stc_lh, stc_rh in zip(stcs_lh, stcs_rh):
            # Determine the parcel label or index for the current source estimate
            coordinates_lh = stc_lh.vertices[0]
            coordinates_rh = stc_rh.vertices[0]

            parcel_label_lh = coordinate_to_parcel_mapping.get(tuple(coordinates_lh), None)
            parcel_label_rh = coordinate_to_parcel_mapping.get(tuple(coordinates_rh), None)

            # Add the source estimate data to the corresponding parcel
            if parcel_label_lh in parcel_data_lh:
                parcel_data_lh[parcel_label_lh].append(stc_lh.data)
            else:
                parcel_data_lh[parcel_label_lh] = [stc_lh.data]

            if parcel_label_rh in parcel_data_rh:
                parcel_data_rh[parcel_label_rh].append(stc_rh.data)
            else:
                parcel_data_rh[parcel_label_rh] = [stc_rh.data]

        # Calculate the average for each parcel for left hemisphere
        for label, data_list in parcel_data_lh.items():
            average_data = np.mean(data_list, axis=0)
            average_stc = mne.SourceEstimate(average_data, vertices=stcs_lh[0].vertices,
                                             tmin=stcs_lh[0].tmin, tstep=stcs_lh[0].tstep,
                                             subject=subject)  # Use 'fsaverage' for population-based template
            average_stcs_batch_lh.append(average_stc)

        # Calculate the average for each parcel for right hemisphere
        for label, data_list in parcel_data_rh.items():
            average_data = np.mean(data_list, axis=0)
            average_stc = mne.SourceEstimate(average_data, vertices=stcs_rh[0].vertices,
                                             tmin=stcs_rh[0].tmin, tstep=stcs_rh[0].tstep,
                                             subject=subject)  # Use 'fsaverage' for population-based template
            average_stcs_batch_rh.append(average_stc)

        # Now you have the average data for each parcel in this batch for both hemispheres
        print(f"Average parcel data for batch {batch_num} (Left Hemisphere):", average_stcs_batch_lh)
        print(f"Average parcel data for batch {batch_num} (Right Hemisphere):", average_stcs_batch_rh)

        # Extract label time courses
        label_tc_lh = mne.extract_label_time_course(average_stcs_batch_lh, labels=labels, src=src, mode='mean_flip')
        label_tc_rh = mne.extract_label_time_course(average_stcs_batch_rh, labels=labels, src=src, mode='mean_flip')

        # Combine label time courses for both hemispheres
        label_time_course_combined = np.concatenate([label_tc_lh, label_tc_rh], axis=1)
        label_time_courses.append(label_time_course_combined)

# Process the remaining files (if any)
remaining_files_lh = inverse_solution_files_lh[len(inverse_solution_files_lh) // batch_size * batch_size:]
remaining_files_rh = inverse_solution_files_rh[len(inverse_solution_files_rh) // batch_size * batch_size:]

if remaining_files_lh and remaining_files_rh:
    # Initialize counters for the remaining files
    remaining_file_num = 1

    for file_path_lh, file_path_rh in zip(remaining_files_lh, remaining_files_rh):
        try:
            stc_epoch_lh = mne.read_source_estimate(file_path_lh)
            stc_epoch_rh = mne.read_source_estimate(file_path_rh)

            # Parcel Averaging for remaining files
            parcel_data_lh = {}
            parcel_data_rh = {}

            for stc_lh, stc_rh in zip(stc_epoch_lh, stc_epoch_rh):
                # Determine the parcel label or index for the current source estimate
                coordinates_lh = stc_lh.vertices[0]
                coordinates_rh = stc_rh.vertices[0]

                parcel_label_lh = coordinate_to_parcel_mapping.get(tuple(coordinates_lh), None)
                parcel_label_rh = coordinate_to_parcel_mapping.get(tuple(coordinates_rh), None)

                # Add the source estimate data to the corresponding parcel
                if parcel_label_lh in parcel_data_lh:
                    parcel_data_lh[parcel_label_lh].append(stc_lh.data)
                else:
                    parcel_data_lh[parcel_label_lh] = [stc_lh.data]

                if parcel_label_rh in parcel_data_rh:
                    parcel_data_rh[parcel_label_rh].append(stc_rh.data)
                else:
                    parcel_data_rh[parcel_label_rh] = [stc_rh.data]

            # Calculate the average for each parcel for left hemisphere
            for label, data_list in parcel_data_lh.items():
                average_data = np.mean(data_list, axis=0)
                average_stc = mne.SourceEstimate(average_data, vertices=stcs_lh[0].vertices,
                                                 tmin=stcs_lh[0].tmin, tstep=stcs_lh[0].tstep,
                                                 subject=subject)  # Use 'fsaverage' for population-based template
                average_stcs_batch_lh.append(average_stc)

            # Calculate the average for each parcel for right hemisphere
            for label, data_list in parcel_data_rh.items():
                average_data = np.mean(data_list, axis=0)
                average_stc = mne.SourceEstimate(average_data, vertices=stcs_rh[0].vertices,
                                                 tmin=stcs_rh[0].tmin, tstep=stcs_rh[0].tstep,
                                                 subject=subject)  # Use 'fsaverage' for population-based template
                average_stcs_batch_rh.append(average_stc)

            # Now you have the average data for each parcel in this batch for both hemispheres
            print(f"Average parcel data for remaining file #{remaining_file_num} - Left Hemisphere:", average_stcs_batch_lh)
            print(f"Average parcel data for remaining file #{remaining_file_num} - Right Hemisphere:", average_stcs_batch_rh)

            # Extract label time courses
            label_tc_lh = mne.extract_label_time_course(average_stcs_batch_lh, labels=labels, src=src, mode='mean_flip')
            label_tc_rh = mne.extract_label_time_course(average_stcs_batch_rh, labels=labels, src=src, mode='mean_flip')

            # Combine label time courses for both hemispheres
            label_time_course_combined = np.concatenate([label_tc_lh, label_tc_rh], axis=1)
            label_time_courses.append(label_time_course_combined)

            # Increment the remaining file number
            remaining_file_num += 1
        except Exception as e:
            print(f"Error processing remaining files {file_path_lh} or {file_path_rh}: {e}")

# Combine the list of label time courses into a single array
label_time_courses_combined = np.concatenate(label_time_courses, axis=1)

# Save the label time courses to a file
output_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy")
np.save(output_file, label_time_courses_combined)

print(f"Label time courses saved to {output_file}")

#################################################################################
# Parcel-to-ROI Mapping
#################################################################################

# Load the Excel file into a DataFrame
brain_region_data = subregions_data

# Assuming average_parcel_data is the dictionary of average source estimate data

# Create a dictionary to store the mapping of parcels to brain regions
parcel_to_brain_region_mapping = {}

# Iterate over the keys (parcel labels) in average_parcel_data
for parcel_label in average_parcel_data.keys():
    # Find the row in the DataFrame that corresponds to the parcel label
    region_row = subregions_data[
        (subregions_data['Label ID.L'] == parcel_label) | (subregions_data['Label ID.R'] == parcel_label)]

    # Check if the region_row is not empty
    if not region_row.empty:
        # Extract the hemisphere information from the 'Left and Right Hemisphere' column
        hemisphere_info = region_row['Left and Right Hemisphere'].values[0]

        # Map the parcel label to the hemisphere info
        parcel_to_brain_region_mapping[parcel_label] = hemisphere_info

# Print the parcel-to-brain-region mapping
print("Parcel-to-Brain-Region Mapping:")
for parcel_label, hemisphere_info in parcel_to_brain_region_mapping.items():
    print(f"Parcel {parcel_label} is in hemisphere: {hemisphere_info}")

#################################################################################
#WILL COME BACK TO THIS PART AFTERWARDS!

# Compute Graph Theory Metrics

# Importing necessary libraries and functions
from scipy.signal import hilbert
import time

# Define the Heaviside step function
def heaviside(x):
    return np.where(x > 0, 1.0, np.where(x == 0, 0.5, 0.0))

# Adapted function to compute dPLI between parcels at the source level
def compute_dPLI_between_parcels(parcel_data1, parcel_data2):
    # Initialize dPLI value
    dpli = 0.5  # Default value for no phase-lead/phase-lag relationship

    # Compute phase difference between parcel_data1 and parcel_data2
    phase_diff = np.angle(parcel_data1 * np.conj(parcel_data2))

    # Compute dPLI using the Heaviside step function
    num_samples = len(phase_diff)
    num_positive = np.sum(heaviside(phase_diff))

    if num_positive > 0:
        dpli = num_positive / num_samples

    return dpli


# Adapted function to compute dPLI at the source level
def compute_dPLI_source_level(average_parcel_data, fmin, fmax):
    print("Starting source-level dPLI computation...")

    dpli_bands = {}
    for i in range(len(fmin)):
        band_name = band_names[i]
        band = f"{fmin[i]}-{fmax[i]}"
        print(f"Processing frequency band: {band_name} - {band}")

        # Initialize dPLI matrix for the current frequency band
        num_parcels = len(average_parcel_data)  # Get the number of parcels
        dpli_matrix = np.empty((num_parcels, num_parcels), dtype=np.float64)

        # Loop through parcels and compute dPLI for each pair
        # Inside the loop that computes dPLI for each pair of parcels
        for parcel_idx1, (parcel_label1, parcel_data1) in enumerate(average_parcel_data.items()):
            for parcel_idx2, (parcel_label2, parcel_data2) in enumerate(average_parcel_data.items()):
                # Debug print to check if data exists for each parcel
                print(f"Parcel {parcel_label1} data: {parcel_data1.shape}")
                print(f"Parcel {parcel_label2} data: {parcel_data2.shape}")
                # Check if parcel_data1 and parcel_data2 are not None
                if parcel_data1 is not None and parcel_data2 is not None:
                    # Compute dPLI for this parcel pair using the Heaviside function-based approach
                    dpli = compute_dPLI_between_parcels(parcel_data1, parcel_data2)
                    dpli_matrix[parcel_idx1, parcel_idx2] = dpli
                else:
                    # Handle cases where parcel_data1 or parcel_data2 is None
                    dpli_matrix[parcel_idx1, parcel_idx2] = 0.5  # Default value

        dpli_bands[band_name] = dpli_matrix

        print(f"Finished source-level dPLI computation for band: {band_name} - {band}")

    print("Finished source-level dPLI computation.")
    return dpli_bands

# Profile the adapted source-level dPLI computation function
start_time = time.time()
result = compute_dPLI_source_level(average_parcel_data, fmin, fmax)
end_time = time.time()
source_level_dpli_time = end_time - start_time
print(f"Source-level dPLI computation execution time: {source_level_dpli_time} seconds")

del result, start_time, end_time, source_level_dpli_time  # Delete the result to free up memory

#################################################################################

# Create a list to store dPLI matrices for each batch
dpli_matrices_bands = []
batch_count = 0

for batch in load_data_in_batches(10):
    if batch is None:
        print("All batches processed.")
        break

    # Extract channel-level data from EEG epochs
    batch_data = np.array([epoch.get_data() for epoch in batch])  # Replace 'get_data()' with the correct method

    # Check which batch is being processed
    print(f"Processing batch {batch_count + 1}...")

    # Logging statement: Check the shape of batch_data (debugging)
    print(f"Batch {batch_count} - Data shape: {batch_data[0].shape}")  # Use [0] to get the shape of a single epoch

    # Reshape to (epochs, channels, time points)
    num_epochs, num_channels, num_time_points = batch_data.shape[0], batch_data.shape[1], batch_data.shape[2] * batch_data.shape[3]
    batch_data = batch_data.reshape((num_epochs, num_channels, num_time_points))

    # Compute dPLI for the batch using the Numba-optimized function
    dpli_matrices_bands.append(compute_dPLI_numba(batch_data, fmin, fmax))
    batch_count += 1

#################################################################################

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

# Print the computed graph metrics for each frequency band --> Should I INCLUDE MODULARITY?
for band in band_names:
    print(f"Frequency Band: {band}")
    print(f"Characteristic Path Length: {L_values[band]}")
    print(f"Global Efficiency: {global_efficiency_values[band]}")
    print(f"Average Directed Clustering Coefficient: {C_values[band]}")
    print(f"Small-Worldness: {sigma_values[band]}")
    print("-" * 50)



#################################################################################



