# Computing and applying a linear minimum-norm inverse method on evoked/raw/epochs data

# Import necessary libraries and functions
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or try 'Qt5Agg', 'Agg', etc.
mne.viz.set_3d_backend("pyvista")
mne.viz.set_3d_backend("notebook")

import mne
from mne.datasets import sample
from mne.minimum_norm import make_inverse_operator, apply_inverse

#################################################################################
# Process EEG data

# Load epoched data
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # --> replace participant ID here
epoched_file = os.path.join(output_dir, subj + '_epoched.fif')
epochs = mne.read_epochs(epoched_file)

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

# Looking at whitened data
evoked.plot_white(noise_cov, time_unit="s")

#################################################################################
# Visualize the source space on the cortex

# Read the forward solution
fwd = mne.read_forward_solution(fwd_fname)
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
trans = mne.read_trans(trans_fname)

# Visualize the cortex
fig = mne.viz.create_3d_figure(size=(600, 400))
mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    trans=trans,
    surfaces="white",
    coord_frame="mri",
    fig=fig,
)

# Mark the position of the dipoles with small red dots
mne.viz.plot_dipole_locations(
    dipoles=dipoles,
    trans=trans,
    mode="sphere",
    subject=subject,
    subjects_dir=subjects_dir,
    coord_frame="mri",
    scale=7e-4,
    fig=fig,
)

mne.viz.set_3d_view(figure=fig, azimuth=180, distance=0.25)

#################################################################################
# Inverse modeling: eLORETA on evoked data with loose dipole orientations

# Create a loose-orientation inverse, with depth weighting
inv = make_inverse_operator(evoked.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=True)

# Compute eLORETA solution
snr = 3.0
lambda2 = 1.0 / snr**2
stc = abs(apply_inverse(evoked, inv, lambda2, "eLORETA", verbose=True))

# Visualization parameters
kwargs = dict(
    initial_time=0.08,
    hemi="lh",
    subjects_dir=subjects_dir,
    size=(600, 600),
    clim=dict(kind="percent", lims=[90, 95, 99]),
    smoothing_steps=7,
    surface="white",  # This ensures the same cortex visualization
    backend="pyvistaqt",  # Use the correct backend
)

# Visualization of eLORETA activations
brain = stc.plot(**kwargs)
brain.add_text(0.1, 0.9, "eLORETA", "title", font_size=14)
del inv

# Visualization of dipole activations
fig, ax = plt.subplots()
ax.plot(1e3 * stc.times, stc.data[::100, :].T)
ax.set(xlabel="time (ms)", ylabel="eLORETA value")

# Plotting original data and residual after fistting
fig, axes = plt.subplots(2, 1)
evoked.plot(axes=axes)
for ax in axes:
    for text in list(ax.texts):
        text.remove()
    for line in ax.lines:
        line.set_color("#98df81")
residual.plot(axes=axes)

