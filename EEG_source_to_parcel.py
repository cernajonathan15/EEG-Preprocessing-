# Source-to-parcel analysis

# Import necessary libraries
import os
import glob
import numpy as np
import pandas as pd
import mne
from mne.datasets import fetch_fsaverage
from nilearn import datasets
from nilearn.image import get_data

# Set your output directory
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # Replace with your subject ID

# Fetch the Schaefer atlas with 100 parcels
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)

# Load the source space for both hemispheres
fs_dir = fetch_fsaverage(verbose=True)
fname = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
src = mne.read_source_spaces(fname, patch_stats=False, verbose=None)

# Load inverse solution file paths for both left and right hemispheres
inverse_solution_files_lh = glob.glob(os.path.join(output_dir, f"{subj}_inversesolution_epoch*.fif-lh.stc"))
inverse_solution_files_rh = glob.glob(os.path.join(output_dir, f"{subj}_inversesolution_epoch*.fif-rh.stc"))

# Initialize a list to store label time courses
label_time_courses = []

# Calculate the total number of inverse solution files for both hemispheres
total_files_lh = len(inverse_solution_files_lh)
total_files_rh = len(inverse_solution_files_rh)

# Calculate the batch size for both hemispheres
batch_size_lh = total_files_lh // (total_files_lh // 10)  # Change 10 to your desired batch size
batch_size_rh = total_files_rh // (total_files_rh // 10)  # Change 10 to your desired batch size

# Ensure batch size is a multiple of 10 (or your desired batch size) for both hemispheres
while total_files_lh % batch_size_lh != 0:
    batch_size_lh -= 1

while total_files_rh % batch_size_rh != 0:
    batch_size_rh -= 1

# Initialize lists to store source estimates for both hemispheres
stcs_lh = []
stcs_rh = []

# Load data in batches for both hemispheres
for i in range(0, total_files_lh, batch_size_lh):
    batch_files_lh = inverse_solution_files_lh[i:i + batch_size_lh]
    batch_files_rh = inverse_solution_files_rh[i:i + batch_size_rh]

    for file_path_lh, file_path_rh in zip(batch_files_lh, batch_files_rh):
        try:
            stc_epoch_lh = mne.read_source_estimate(file_path_lh)
            stc_epoch_rh = mne.read_source_estimate(file_path_rh)
            stcs_lh.append(stc_epoch_lh)
            stcs_rh.append(stc_epoch_rh)
        except Exception as e:
            print(f"Error loading files {file_path_lh} or {file_path_rh}: {e}")

# Load labels from the atlas
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', subjects_dir=r'C:\Users\cerna\mne_data\MNE-fsaverage-data')

# Extract label time courses for both hemispheres
for idx, (stc_lh, stc_rh) in enumerate(zip(stcs_lh, stcs_rh)):
    try:
        label_tc_lh = stc_lh.extract_label_time_course(labels, src=src, mode='mean_flip')
        label_tc_rh = stc_rh.extract_label_time_course(labels, src=src, mode='mean_flip')
        label_time_courses.extend([label_tc_lh, label_tc_rh])
    except Exception as e:
        print(f"Error extracting label time courses for iteration {idx}: {e}")
else:  # This block will execute if the for loop completes without encountering a break statement
    print("All time courses have been successfully extracted!")

# Convert label_time_courses to a NumPy array
label_time_courses_np = np.array(label_time_courses)

# If you prefer to save as a .csv file
# Convert to DataFrame and save as .csv
#label_time_courses_df = pd.DataFrame(label_time_courses_np)
#label_time_courses_df.to_csv(os.path.join(output_dir, f"{subj}_label_time_courses.csv"), index=False)

# Save the label time courses as a .npy file
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
label_time_courses_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy")
np.save(label_time_courses_file, label_time_courses_np)

#######################################################################################################################

# Plotting Time Courses

import matplotlib
matplotlib.use('Qt5Agg')  # or 'Agg' if you're running non-interactively
import matplotlib.pyplot as plt

# Choose a random time course to plot
random_idx = np.random.randint(len(label_time_courses))
random_time_course = label_time_courses[random_idx]

# Plot the time course
plt.figure(figsize=(10, 6))
plt.plot(random_time_course)
plt.title(f'Time Course for Randomly Selected Region: {random_idx}')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Connectivity Visualization

from scipy.stats import pearsonr

# Compute a connectivity matrix (e.g., using Pearson correlation)
num_regions = len(label_time_courses[0])
connectivity_matrix = np.zeros((num_regions, num_regions))

for i in range(num_regions):
    for j in range(num_regions):
        connectivity_matrix[i, j], _ = pearsonr(label_time_courses[0][i], label_time_courses[0][j])

# Visualize the connectivity matrix
plt.figure(figsize=(10, 10))
plt.imshow(connectivity_matrix, cmap='viridis', origin='lower')
plt.title('Connectivity Matrix')
plt.xlabel('Region')
plt.ylabel('Region')
plt.colorbar(label='Pearson Correlation')
plt.show()

#######################################################################################################################


