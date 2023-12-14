
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

src = mne.read_source_spaces(fname, patch_stats=False, verbose=None)

# Initialize a list to store label time courses
label_time_courses = []

# Define labels based on the Brainnetome atlas for both hemispheres
labels = []

for index, row in subregions_data.iterrows():
    # Create labels for the left and right hemispheres
    label_left = f"Left_{label_l}"
    label_right = f"Right_{label_r}"

    # Add both hemisphere labels to the list
    labels.extend([label_left, label_right])

# Initialize a list to store average_stc for each batch
average_stcs_lh = []
average_stcs_rh = []

# Calculate the total number of inverse solution files
total_files = len(inverse_solution_files_lh)

# Calculate the batch size to process all files without remaining files
batch_size = total_files // (total_files // 10)  # Change 10 to your desired batch size

# Ensure batch size is a multiple of 10 (or your desired batch size)
while total_files % batch_size != 0:
    batch_size -= 1

# Initialize empty lists to store the labels for left and right hemispheres
parcel_labels_lh = []
parcel_labels_rh = []

# Load data in batches
for i in range(0, total_files, batch_size):
    batch_files_lh = inverse_solution_files_lh[i:i + batch_size]
    batch_files_rh = inverse_solution_files_rh[i:i + batch_size]

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

            # Append the labels to the respective lists
            parcel_labels_lh.append(parcel_label_lh)
            parcel_labels_rh.append(parcel_label_rh)

        # Calculate the average for each parcel for left hemisphere
        average_stcs_batch_lh = {}  # Initialize a dictionary to store average_stc objects for this batch
        for label, data_list in parcel_data_lh.items():
            average_data = np.mean(data_list, axis=0)
            average_stc = mne.SourceEstimate(average_data, vertices=stcs_lh[0].vertices,
                                             tmin=stcs_lh[0].tmin, tstep=stcs_lh[0].tstep,
                                             subject=stcs_lh[0].subject)
            average_stcs_batch_lh[label] = average_stc

        # Calculate the average for each parcel for right hemisphere
        average_stcs_batch_rh = {}  # Initialize a dictionary to store average_stc objects for this batch
        for label, data_list in parcel_data_rh.items():
            average_data = np.mean(data_list, axis=0)
            average_stc = mne.SourceEstimate(average_data, vertices=stcs_rh[0].vertices,
                                             tmin=stcs_rh[0].tmin, tstep=stcs_rh[0].tstep,
                                             subject=stcs_rh[0].subject)
            average_stcs_batch_rh[label] = average_stc

        # Now you have the average data for each parcel in this batch for both hemispheres
        print(f"Average parcel data for batch {i // batch_size} (Left Hemisphere):", average_stcs_batch_lh)
        print(f"Average parcel data for batch {i // batch_size} (Right Hemisphere):", average_stcs_batch_rh)

        # Store the average STCs for each hemisphere separately
        average_stcs_lh.extend(list(average_stcs_batch_lh.values()))
        average_stcs_rh.extend(list(average_stcs_batch_rh.values()))

# Concatenate the list of SourceEstimate objects into a single SourceEstimate for both hemispheres
average_stc_combined_lh = mne.compute_source_morph(average_stcs_lh[0], subject_from='fsaverage', subject_to='fsaverage')
average_stc_combined_rh = mne.compute_source_morph(average_stcs_rh[0], subject_from='fsaverage', subject_to='fsaverage')

############################################################################################################

# Import the necessary libraries
from mne.transforms import apply_trans
from mne.datasets import fetch_fsaverage
from mne.source_space import SourceSpaces

# Load subject directory
subjects_dir = output_dir

# Load the transformation matrix to MNI space
trans = mne.read_trans(r'C:\Users\cerna\Downloads\101_inverse_operator.fif')

# Create an empty list to store transformed source estimates
transformed_stcs = []

# Iterate through your average_stcs_lh and average_stcs_rh lists
for stc_lh, stc_rh in zip(average_stcs_lh, average_stcs_rh):
    # Apply the transformation to the left hemisphere source estimate
    stc_lh_mni = apply_trans(trans, stc_lh)

    # Apply the transformation to the right hemisphere source estimate
    stc_rh_mni = apply_trans(trans, stc_rh)

    # Add the transformed source estimates to the list
    transformed_stcs.append(stc_lh_mni)
    transformed_stcs.append(stc_rh_mni)

# Now, the 'transformed_stcs' list contains your source estimates aligned with the Brainnetome atlas in MNI space.

#####################################################################################################################

# OPTIONAL VISUALIZATION: Left and right hemisphere visualization of the average source estimate at a specific time point

# Assuming you have already computed the average STCs and loaded the Brainnetome atlas data as mentioned in your code

# Create an empty list to store the labels corresponding to the source estimates
#source_labels = []

# Create an empty list to store the vertices for left and right hemispheres
#lh_vertices = []
#rh_vertices = []

# Create a list to store the time points for the source estimates
#time_points = []

# Define subjects_dir
#subjects_dir = output_dir  # Replace with your actual subjects_dir path

# Iterate through your average_stcs_lh and average_stcs_rh lists
#for stc_lh, stc_rh in zip(average_stcs_lh, average_stcs_rh):
    # Combine the left and right hemisphere source estimates
#    stc_combined = mne.SourceEstimate(stc_lh.data + stc_rh.data, stc_lh.vertices, tmin=stc_lh.tmin, tstep=stc_lh.tstep, subject='fsaverage')

    # Append the labels corresponding to this source estimate
#    source_labels.append(stc_combined)

    # Append the vertices for left and right hemispheres
#    lh_vertices.append(stc_lh.vertices)
#    rh_vertices.append(stc_rh.vertices)

    # Append the time point
#    time_points.append(stc_combined.times[0])  # You can choose a specific time point here

# Choose a specific time point for visualization (e.g., the first time point)
#time_index = 0  # Change this index to the desired time point

# Visualize the average source estimate at the chosen time point
#stc_combined_at_time = source_labels[time_index]
#stc_combined_at_time.plot(
#    subject='fsaverage',
#    hemi='split',
#    views=['lateral', 'medial'],
#    initial_time=time_points[time_index],  # Set to the chosen time point
#    subjects_dir=subjects_dir,
#    clim=dict(kind="percent", lims=[90, 95, 99]),  # Adjust clim as needed
#    colormap="viridis",  # Choose an appropriate colormap
#    title=f'Source Estimate at {time_points[time_index] * 1000:.1f} ms',  # Include the time in milliseconds
#    background="white",  # Adjust background color
#    time_unit='s'  # Units for time
#)

#####################################################################################################################

# Link regions with networks

# Load the Brainnetome subregions Excel file into a DataFrame
subregions_func_network_file = os.path.join(output_dir, "subregion_func_network_Yeo_updated.csv")
subregions_func_network_data = pd.read_csv(subregions_func_network_file)

# Create an empty list to store DataFrames to be concatenated later
correspondence_dfs = []

# Iterate through your subregions_data DataFrame
for index, row in subregions_data.iterrows():
    # Split the "Left and Right Hemisphere" column by semicolons
    hemisphere_rows = row['Left and Right Hemisphere'].split(';')

    for hemisphere_row in hemisphere_rows:
        hemisphere_parts = hemisphere_row.split('_')
        if len(hemisphere_parts) != 4:
            # Handle cases where the format is not as expected
            continue

        hemisphere_left = f"{hemisphere_parts[0]}_{hemisphere_parts[1][0]}_{hemisphere_parts[3]}"
        hemisphere_right = f"{hemisphere_parts[0]}_{hemisphere_parts[2][0]}_{hemisphere_parts[3]}"

        # Now, fetch the corresponding information from subregions_func_network_data DataFrame
        subregion_info_left = subregions_func_network_data[
            subregions_func_network_data['region'] == hemisphere_left
            ]
        subregion_info_right = subregions_func_network_data[
            subregions_func_network_data['region'] == hemisphere_right
            ]

        # Assuming there's a unique match for left and right hemispheres
        subregion_info_left = subregion_info_left.iloc[0] if not subregion_info_left.empty else None
        subregion_info_right = subregion_info_right.iloc[0] if not subregion_info_right.empty else None

        # Extract the information from the subregions_func_network_data DataFrame for left hemisphere
        label_id_l = subregion_info_left['Label ID.L'] if subregion_info_left is not None else ''
        label_id_r = subregion_info_right['Label ID.R'] if subregion_info_right is not None else ''
        yeo_7network = subregion_info_left['Yeo 7 Network Membership'] if subregion_info_left is not None else ''
        yeo_17network = subregion_info_left['Yeo 17 Network Membership'] if subregion_info_left is not None else ''

        # Create DataFrames for left and right hemispheres and add them to the list
        left_df = pd.DataFrame(
            {'Label ID.L': [label_id_l], 'Label ID.R': [label_id_r], 'Yeo 7 Network Membership': [yeo_7network],
             'Yeo 17 Network Membership': [yeo_17network], 'region': [hemisphere_left]})
        right_df = pd.DataFrame(
            {'Label ID.L': [label_id_l], 'Label ID.R': [label_id_r], 'Yeo 7 Network Membership': [yeo_7network],
             'Yeo 17 Network Membership': [yeo_17network], 'region': [hemisphere_right]})

        correspondence_dfs.extend([left_df, right_df])

# Concatenate all DataFrames in the list into one DataFrame
correspondence_df = pd.concat(correspondence_dfs, ignore_index=True)

# Now, correspondence_df contains the correspondence you need

# Assuming you have transformed source estimates stored in transformed_stcs
# First, create a list to store the labels corresponding to the source estimates
source_labels = []

# Iterate through your transformed source estimates
for stc in transformed_stcs:
    # Determine the label for the current source estimate based on its subject and hemisphere
    hemisphere = 'Left' if 'lh' in stc.subject else 'Right'
    label = f"{hemisphere}_{stc.subject.split('_')[-1]}"  # Extract the label ID

    # Append the label to the source_labels list
    source_labels.append(label)

# Add the source_labels list as a new column to the correspondence_df DataFrame
correspondence_df['Label'] = source_labels

# Now, correspondence_df contains the correspondence between your source estimates and the desired structure

















