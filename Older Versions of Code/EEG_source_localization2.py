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
channels_to_drop = ['LHEye', 'RHEye', 'Lneck', 'Rneck', 'RVEye']

# Drop the channels from the epochs data
epochs.drop_channels(channels_to_drop)

# Adjust EEG electrode locations to match the fsaverage template
montage_path = r"C:\Users\cerna\Downloads\MATLAB\MFPRL_UPDATED_V2.sfp"
montage = mne.channels.read_custom_montage(montage_path)


montage = mne.channels.make_standard_montage("standard_1005") # change to the custome montage I'm using
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

# Compute the evoked response
evoked = epochs.average().pick(picks)
evoked.plot(time_unit="s")
evoked.plot_topomap(times=np.linspace(0.05, 0.15, 5), ch_type="eeg")

# Looking at whitened data to see if the noise is Gaussian
evoked.plot_white(noise_cov, time_unit="s")

#################################################################################
# Visualize the source space on the cortex

# Read the forward solution
mne.convert_forward_solution(fwd, surf_ori=True, copy=False)

# Extract the source space information from the forward solution
lh = fwd["src"][0]  # Visualize the left hemisphere
verts = lh["rr"]  # The vertices of the source space
tris = lh["tris"]  # Groups of three vertices that form triangles
dip_pos = lh["rr"][lh["vertno"]]  # The position of the dipoles
dip_ori = lh["nn"][lh["vertno"]]
dip_len = len(dip_pos)
dip_times = [0]

# Create a Dipole instance
actual_amp = np.ones(dip_len)  # misc amp to create Dipole instance
actual_gof = np.ones(dip_len)  # misc GOF to create Dipole instance
dipoles = mne.Dipole(dip_times, dip_pos, actual_amp, dip_ori, actual_gof)
trans = trans

# Create a new 3D figure and plot red dots at the dipole locations (run all code below at once)
fig = mne.viz.create_3d_figure(size=(600, 400))
mne.viz.plot_alignment( # Plot the cortical surface on the figure
    subject=subject,
    subjects_dir=subjects_dir,
    trans=trans,
    surfaces="white",
    coord_frame="mri",
    fig=fig,
)
mne.viz.plot_dipole_locations( # Plot the dipoles on the same figure
    dipoles=dipoles,
    trans=trans,
    mode="sphere",
    subject=subject,
    subjects_dir=subjects_dir,
    coord_frame="mri",
    scale=7e-4,
    fig=fig,
)
mne.viz.set_3d_view(figure=fig, azimuth=180, distance=0.25) # Adjust the view

# Save the computed forward solution to a .fif file
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
forward_solution_file = os.path.join(output_dir, subj + '_forwardsolution_MRItemplate.fif')
mne.write_forward_solution(forward_solution_file, fwd, overwrite=True)

#################################################################################
# Inverse modeling: eLORETA on evoked data with loose dipole orientations

# Create a loose-orientation inverse operator, with depth weighting
inv = make_inverse_operator(evoked.info, fwd, noise_cov, fixed=False, loose=0.2, depth=0.8, verbose=True)

# Compute eLORETA solution
snr = 3.0
lambda2 = 1.0 / snr**2
stc = abs(apply_inverse(evoked, inv, lambda2, "eLORETA", verbose=True, pick_ori="vector"))

# Visualization parameters
kwargs = dict(
    initial_time=0.08,
    hemi="lh",
    subjects_dir=subjects_dir,
    size=(600, 600),
    clim=dict(kind="percent", lims=[90, 95, 99]),
    smoothing_steps=7,
)

# Visualizing source estimate with loose dipole orientations at peak activity time point (run all code below at once)
_, time_max = stc.magnitude().get_peak(hemi="lh")
brain_loose = stc.plot(
    subjects_dir=subjects_dir,
    initial_time=time_max,
    time_unit="s",
    size=(600, 400),
    overlay_alpha=0,
)
mne.viz.set_3d_view(figure=brain_loose, focalpoint=(0.0, 0.0, 50))

# Visualization of eLORETA activations
brain = stc.plot(**kwargs)
brain.add_text(0.1, 0.9, "eLORETA", "title", font_size=14)
brain.show_view()

# Average the data across the second dimension
avg_data = stc.data[::100, :].mean(axis=1).T

# Visualization of dipole activations
fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, avg_data)
ax.set(xlabel="time (ms)", ylabel="eLORETA value (average)")
plt.show()

# Save the inverse solution data --> need to understand why this file is not being saved
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
inverse_solution_file = os.path.join(output_dir, subj + '_inversesolution.fif')
stc.save(inverse_solution_file)



