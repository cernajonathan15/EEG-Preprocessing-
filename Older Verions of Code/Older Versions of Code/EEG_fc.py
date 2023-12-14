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
labels_dir = os.path.join(subjects_dir, 'fsaverage', 'label')
os.makedirs(labels_dir, exist_ok=True)

base_url = "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/c773720ad340dcb1d566b0b8de68b6acdf2ca505/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage/label"

# Generate file names for desired parcel counts and hemispheres
parcel_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
hemispheres = ['lh', 'rh']
files = [f"{hemi}.Schaefer2018_{count}Parcels_7Networks_order.annot" for count in parcel_counts for hemi in hemispheres]

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
# Set needed directories and load data

# Import necessary libraries and functions
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import mne

# Setting directories and file names
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # --> replace participant ID here

# Load and average the computed inverse solution data across epochs
stcs = []  # List to store source estimates for each epoch
for epoch_num in range(58):  # Adjust the range if the number of epochs changes
    inverse_solution_file = os.path.join(output_dir, f"{subj}_inversesolution_epoch{epoch_num}.fif-stc")
    stc_epoch = mne.read_source_estimate(inverse_solution_file)
    stcs.append(stc_epoch)

# Average the source estimates across epochs
stc_avg = np.mean(stcs, axis=0) # Now, stc_avg contains the averaged source estimate across all epochs

# Compute the standard deviation across epochs for each source space point
stc_std = np.std(stcs, axis=0)

# Plot the variability map on the brain
# Note: This will open an interactive 3D visualization. Ensure you have a suitable backend for visualization.
brain_std = stc_std.plot(hemi='both', subjects_dir=subjects_dir, clim='auto', colormap='viridis', surface='inflated')

#################################################################################

#Might need to change the atlas that I am using... need one with cortical and subcortical regions (Brainnetome, AAL or HCP-MMP1.0)
# Assuming you have the Schaefer atlas labels saved in the 'labels' directory for 'fsaverage'
schaefer_dir = os.path.join(subjects_dir, 'fsaverage', 'label', 'Schaefer2018')  # Adjust the path if needed
n_parcels = 400  # Choose the desired resolution (e.g., 100, 200, 300, 400, ...)

# Load Schaefer's atlas labels. This assumes labels are named in a format like 'lh.Schaefer2018_400Parcels_17Networks_order.label'
labels = [mne.read_label(os.path.join(schaefer_dir, hemi + f'.Schaefer2018_{n_parcels}Parcels_17Networks_order.label')) for hemi in ['lh', 'rh'] for i in range(1, n_parcels+1)]

# Exclude medial wall if necessary
labels = [label for label in labels if label.name not in ['lh.Medial_wall', 'rh.Medial_wall']]

# Extract time series for each label
stcs = apply_inverse_epochs(epochs, inv, lambda2, method="eLORETA", pick_ori="vector", return_generator=True)
label_ts = mne.extract_label_time_course(stcs, labels, inv, return_generator=True)
