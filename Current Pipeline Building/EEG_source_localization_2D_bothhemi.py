# Computing and applying a linear minimum-norm inverse method on evoked/raw/epochs data

# Import necessary libraries and functions
import os
import os.path as op
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # Setting the backend BEFORE importing pyplot
import matplotlib.pyplot as plt

import mne
mne.viz.set_3d_backend("pyvista")
from mne.datasets import sample, eegbci, fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse

#################################################################################
# Adult template MRI (fsaverage)

# Download fsaverage files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# The files live in:
subject = "fsaverage"
trans = "fsaverage"  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")
bem = op.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

#################################################################################

# Process EEG data

# Load epoched data
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # --> replace participant ID here
epoched_file = os.path.join(output_dir, subj + '_epoched.fif')
epochs = mne.read_epochs(epoched_file)

# List of channels to drop
channels_to_drop = ['LHEye', 'RHEye', 'Lneck', 'Rneck', 'RVEye', 'FPz']

# Drop the channels from the epochs data
epochs.drop_channels(channels_to_drop)

# Adjust EEG electrode locations to match the fsaverage template, which are already in fsaverage's
# # space (MNI space) for standard_1020
montage_path = r"C:\Users\cerna\Downloads\MATLAB\MFPRL_UPDATED_V2.sfp"
montage = mne.channels.read_custom_montage(montage_path)
epochs.set_montage(montage)
epochs.set_eeg_reference(projection=True)  # needed for inverse modeling

# Set the 3D backend to pyvista
mne.viz.set_3d_backend("pyvista")

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    epochs.info,
    src=src,
    eeg=["original", "projected"],
    trans=trans,
    show_axes=True,
    mri_fiducials=True,
    dig="fiducials",
)

# Compute the forward solution using the fsaverage template
fwd = mne.make_forward_solution(
    epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None
)

# Adjusting picks to EEG data
picks = mne.pick_types(epochs.info, meg=False, eeg=True, eog=True, stim=False)

# Compute regularized noise covariance
noise_cov = mne.compute_covariance(
    epochs, tmax=0.0, method=["shrunk", "empirical"], rank=None, verbose=True
)

fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info)

#################################################################################
# Visualize the source space on the cortex

# Read the forward solution
mne.convert_forward_solution(fwd, surf_ori=True, copy=False)

# Extract the source space information from the forward solution for both hemispheres
lh = fwd['src'][0]  # Left hemisphere source space information
rh = fwd['src'][1]  # Right hemisphere source space information

# Since lh['vertno'] and rh['vertno'] are arrays of vertex indices for the left and right hemisphere,
# respectively, where dipoles are located, we need to define the dipole times and dummy amplitudes for both.

# Define dummy variables for dipole times and goodness-of-fit for illustration purposes
dip_times = [0]  # Assuming we have a single time point for the example
actual_amp = 100e-9  # Example amplitude in nanoAmperes
actual_gof = 99  # Example goodness-of-fit in percentage

# Create Dipole instances for both hemispheres
lh_dip_pos = lh['rr'][lh['vertno']]  # The position of the dipoles in the left hemisphere
lh_dip_ori = lh['nn'][lh['vertno']]  # The orientation of the dipoles in the left hemisphere
rh_dip_pos = rh['rr'][rh['vertno']]  # The position of the dipoles in the right hemisphere
rh_dip_ori = rh['nn'][rh['vertno']]  # The orientation of the dipoles in the right hemisphere

# Dipole instance creation for left hemisphere
lh_dipoles = mne.Dipole(dip_times, lh_dip_pos, actual_amp * np.ones(len(lh['vertno'])),
                         lh_dip_ori, actual_gof * np.ones(len(lh['vertno'])))

# Dipole instance creation for right hemisphere
rh_dipoles = mne.Dipole(dip_times, rh_dip_pos, actual_amp * np.ones(len(rh['vertno'])),
                         rh_dip_ori, actual_gof * np.ones(len(rh['vertno'])))

# Visualize the source space on the cortex for both hemispheres (run up to mne.viz.set_3d_view(figure=fig, azimuth=180, distance=0.25) at once)
fig = mne.viz.create_3d_figure(size=(600, 400))
mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    trans=trans,
    surfaces="white",
    coord_frame="mri",
    fig=fig,
)
# Plot left hemisphere dipoles
mne.viz.plot_dipole_locations(
    lh_dipoles,
    trans=trans,
    mode="sphere",
    subject=subject,
    subjects_dir=subjects_dir,
    coord_frame="mri",
    scale=1e-3,
    fig=fig,
)
# Plot right hemisphere dipoles
mne.viz.plot_dipole_locations(
    rh_dipoles,
    trans=trans,
    mode="sphere",
    subject=subject,
    subjects_dir=subjects_dir,
    coord_frame="mri",
    scale=1e-3,
    fig=fig,
)
# Adjust the view
mne.viz.set_3d_view(figure=fig, azimuth=180, distance=0.25)

# Save the figure if needed
#fig.savefig(os.path.join(output_dir, f"{subj}_dipole_locations.png"))

# Save the computed forward solution to a .fif file
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
forward_solution_file = os.path.join(output_dir, subj + '_forwardsolution_MRItemplate.fif')
mne.write_forward_solution(forward_solution_file, fwd, overwrite=True)

#################################################################################
# Inverse modeling: eLORETA on evoked data with dipole orientation discarded (pick_ori="None"), only magnitude kept

from mne.minimum_norm import apply_inverse_epochs

# Create a loose-orientation inverse operator, with depth weighting
inv = make_inverse_operator(epochs.info, fwd, noise_cov, fixed=False, loose=0.2, depth=0.8, verbose=True)

# Compute eLORETA solution for each epoch
snr = 3.0
lambda2 = 1.0 / snr**2
stcs = apply_inverse_epochs(epochs, inv, lambda2, "eLORETA", verbose=True, pick_ori=None) # pick_ori="None" --> Discard dipole orientation, only keep magnitude

# Average the source estimates across epochs
stc_avg = sum(stcs) / len(stcs)

# Get the time of the peak magnitude across both hemispheres
peak_hemi, time_max = stc_avg.get_peak()

# Visualization parameters for plotting without specifying a hemisphere
kwargs = dict(
    subjects_dir=subjects_dir,
    size=(600, 600),
    clim=dict(kind="percent", lims=[90, 95, 99]),
    smoothing_steps=7,
    time_unit="s",
    initial_time=time_max  # Set the initial_time to the time of the peak magnitude
)

# Visualizing the averaged source estimate with dipole magnitude across both hemispheres
brain_magnitude = stc_avg.plot(**kwargs)
mne.viz.set_3d_view(figure=brain_magnitude, focalpoint=(0.0, 0.0, 50))

# Average the data across all source space points
avg_data = stc_avg.data.mean(axis=(0, 1))

# Plot the average data as a function of time
fig, ax = plt.subplots()
ax.plot(1e3 * stc_avg.times, avg_data)
ax.set(xlabel="time (ms)", ylabel="eLORETA value (average)")
plt.show()

# Save the inverse solution data
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
for idx, stc in enumerate(stcs):
    inverse_solution_file = os.path.join(output_dir, f"{subj}_inversesolution_epoch{idx}.fif")
    stc.save(inverse_solution_file, overwrite=True)



