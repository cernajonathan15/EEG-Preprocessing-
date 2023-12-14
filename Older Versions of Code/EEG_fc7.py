
# Import libraries
from nilearn import datasets
import glob
import mne
import os
from mne.datasets import fetch_fsaverage
import os.path as op

# Open Schaeffer Atlas
atlas = datasets.fetch_atlas_schaefer_2018(100)

# Setting directories and file names
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # --> replace participant ID here
fs_dir = fetch_fsaverage(verbose=True)

# Load inverse solution files
inverse_solution_files = glob.glob(os.path.join(output_dir, f"{subj}_inversesolution_epoch*.fif-stc.h5"))
fname = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")

src = mne.read_source_spaces(fname, patch_stats=False, verbose=None)
# Batch size
batch_size = 10

# Load data in batches
for i in range(0, len(inverse_solution_files), batch_size):
    batch_files = inverse_solution_files[i:i + batch_size]
    stcs = []

    for file_path in batch_files:
        try:
            stc_epoch = mne.read_source_estimate(file_path)
            stcs.append(stc_epoch)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")


labels = r"C:\Users\cerna\nilearn_data\schaefer_2018\Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
rois = stcs[0].extract_label_time_course(labels=labels, src=src) # keep working on extract_label_time_course function
