# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 12:58:18 2023

@author: cerna
"""

#################################################################################

# PREPROCESSING EEG DATA

#################################################################################

# Import necessary Python modules
import os
import numpy as np
import mne
from scipy.stats import zscore
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg', depending on your system
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import FastICA, PCA
from scipy import signal
#BELOW IS AN EXAMPLE - REPLACE WITH REAL DATA
# Source of adapted code below (reference if needed):
# https://iq.opengenus.org/eeg-signal-analysis-with-python/

#################################################################################

# Define the path to your EEG data
subj_data_path = r'C:\Users\cerna\Downloads'   #replace with your directory where you saved EEG data
fname = 'TCOA_101_EC'  # --> replace file name here
subj = '101'  # --> replace participant ID here
file_path = os.path.join(subj_data_path, fname + '.vhdr')

# Print the file path to check if it's correct
print(f'Attempting to load file at: {file_path}')

# Load data
EEG = mne.io.read_raw_brainvision(file_path, preload=True)

# Check the number of channels in your EEG data by examining the structure
print(EEG.ch_names)

# Drop last 8 channels (Aux channels)
channels_to_drop = EEG.ch_names[-8:]
EEG.drop_channels(channels_to_drop)

# Plot the data to visualize waveforms (all remaining channels)
EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# Checking data attributes to make sure we have the right number of channels
print(EEG.info)


#################################################################################

# Old procedure: Re-referencing to average of mastoids

# Plot the data to visualize waveforms before re-referencing
#EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# Re-referencing to average of mastoids
#these two reference channels will be automatically dropped out for ICA
#EEG.set_eeg_reference(ref_channels=['Ch10', 'Ch21'])  #Green 10 = Left mastoid; Green 21 = Right mastoid

# Plot the data to visualize waveforms after re-referencing
#EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# Plotting EEG signal via PSD to identify potential noise that needs to filtered
#EEG.plot_psd()

#################################################################################

# Read the custom montage
montage_path = r"C:\Users\cerna\Downloads\MATLAB\MFPRL_UPDATED_V2.sfp"
montage = mne.channels.read_custom_montage(montage_path)

# Define the map of channel names using the provided keys
ch_map = {'Ch1': 'Fp1', 'Ch2': 'Fz', 'Ch3': 'F3', 'Ch4': 'F7', 'Ch5': 'LHEye', 'Ch6': 'FC5',
          'Ch7': 'FC1', 'Ch8': 'C3', 'Ch9': 'T7', 'Ch10': 'GND', 'Ch11': 'CP5', 'Ch12': 'CP1',
          'Ch13': 'Pz', 'Ch14': 'P3', 'Ch15': 'P7', 'Ch16': 'O1', 'Ch17': 'Oz', 'Ch18': 'O2',
          'Ch19': 'P4', 'Ch20': 'P8', 'Ch21': 'Rmastoid', 'Ch22': 'CP6', 'Ch23': 'CP2', 'Ch24': 'Cz',
          'Ch25': 'C4', 'Ch26': 'T8', 'Ch27': 'RHEye', 'Ch28': 'FC6', 'Ch29': 'FC2', 'Ch30': 'F4',
          'Ch31': 'F8', 'Ch32': 'Fp2', 'Ch33': 'AF7', 'Ch34': 'AF3', 'Ch35': 'AFz', 'Ch36': 'F1',
          'Ch37': 'F5', 'Ch38': 'FT7', 'Ch39': 'FC3', 'Ch40': 'FCz', 'Ch41': 'C1', 'Ch42': 'C5',
          'Ch43': 'TP7', 'Ch44': 'CP3', 'Ch45': 'P1', 'Ch46': 'P5', 'Ch47': 'Lneck', 'Ch48': 'PO3',
          'Ch49': 'POz', 'Ch50': 'PO4', 'Ch51': 'Rneck', 'Ch52': 'P6', 'Ch53': 'P2', 'Ch54': 'CPz',
          'Ch55': 'CP4', 'Ch56': 'TP8', 'Ch57': 'C6', 'Ch58': 'C2', 'Ch59': 'FC4', 'Ch60': 'FT8',
          'Ch61': 'F6', 'Ch62': 'F2', 'Ch63': 'AF4', 'Ch64': 'RVEye'}

# Rename the channels using the new ch_map
EEG.rename_channels(ch_map)

# Now the channels should match the names in the montage
EEG.set_montage(montage, on_missing='warn')

# High-pass filter (remove anything below 0.1 Hz)
EEG.filter(0.1, None)

# Low-pass filter (remove anything above 50 Hz)
EEG.filter(None, 50.)

# Add a notch filter from 60 Hz
freqs = np.arange(60, 241, 60)  # This will create an array [60, 120, 180, 240] to capture the harmonics
EEG.notch_filter(freqs)

# Plot the data to visualize waveforms after filtering
EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# Plotting EEG signal via PSD to check if the notch filter removed the power line noise
EEG.plot_psd()

# Save the filtered data
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
preprocessed_file = os.path.join(output_dir, subj + '_filtered.fif')
EEG.save(preprocessed_file, overwrite=True)

#################################################################################

# MARKING BAD CHANNELS

import copy # This is a Python module that allows you to copy objects without changing the original object

# Visualize all channels to identify bad channels
EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# This can be used to plot the data with the bad channels marked.
# Uncomment the two lines of code below to see the plot
picks = mne.pick_channels_regexp(EEG.ch_names, regexp="AF.")  # Replace 'regexp=" ."' with the tentative bad channels
EEG.plot(order=picks, n_channels=len(picks))

# Change list of bad channels
original_bads = copy.deepcopy(EEG.info["bads"])
EEG.info["bads"].append("AF7")  # add a single channel
#raw.info["bads"].extend(["EEG 051", "EEG 052"])  # add a list of channels

# Print the bad channels to double check
print(EEG.info['bads'])

#############################-ONLY RUN NEXT STEP IF BAD CHANNELS ARE FOUND TO INTERPOLATE-#############################

# Interpolate bad channels

# Keep a reference to the original, uncropped data
original_EEG = EEG

# Crop a copy of the data to three seconds for easier plotting
cropped_EEG = EEG.copy().crop(tmin=0, tmax=3).load_data()

# Replace NaN or inf values in channel locations with zero
new_chs = original_EEG.info['chs'].copy()
for ch in new_chs:
    ch['loc'] = np.nan_to_num(ch['loc'], nan=0.0, posinf=0.0, neginf=0.0)

new_info = mne.create_info([ch['ch_name'] for ch in new_chs], original_EEG.info['sfreq'], ch_types='eeg')
original_EEG = mne.io.RawArray(original_EEG.get_data(), new_info)
original_EEG.set_montage(mne.channels.make_dig_montage(ch_pos={ch['ch_name']: ch['loc'][:3] for ch in new_chs}))
original_EEG.info['bads'] = original_bads  # Set the bad channels back to the original list

# Repeat for cropped_EEG
new_chs = cropped_EEG.info['chs'].copy()
for ch in new_chs:
    ch['loc'] = np.nan_to_num(ch['loc'], nan=0.0, posinf=0.0, neginf=0.0)

new_info = mne.create_info([ch['ch_name'] for ch in new_chs], cropped_EEG.info['sfreq'], ch_types='eeg')
cropped_EEG = mne.io.RawArray(cropped_EEG.get_data(), new_info)
cropped_EEG.set_montage(mne.channels.make_dig_montage(ch_pos={ch['ch_name']: ch['loc'][:3] for ch in new_chs}))
cropped_EEG.info['bads'] = original_bads  # Set the bad channels back to the original list

# Pick types and interpolate bads
original_EEG_data = original_EEG.copy().pick_types(meg=False, eeg=True, exclude=[])
original_EEG_data_interp = original_EEG_data.copy().interpolate_bads(reset_bads=False)

cropped_EEG_data = cropped_EEG.copy().pick_types(meg=False, eeg=True, exclude=[])
cropped_EEG_data_interp = cropped_EEG_data.copy().interpolate_bads(reset_bads=False)

# Plot the data before and after interpolation
for title, data in zip(["cropped orig.", "cropped interp."], [cropped_EEG_data, cropped_EEG_data_interp]):
    with mne.viz.use_browser_backend("matplotlib"):
        fig = data.plot(butterfly=True, color="#00000022", bad_color="r")
    fig.subplots_adjust(top=0.9)
    fig.suptitle(title, size="xx-large", weight="bold")

# Save the interpolated data
output_dir = r'C:\Users\cerna\Downloads'
preprocessed_file = os.path.join(output_dir, subj + '_interpolated.fif')
original_EEG_data_interp.save(preprocessed_file, overwrite=True)

#################################################################################

# ICA (Independent Component Analysis) --> PICK IT UP FROM HERE! Replace with interpolated data now....

from mne.preprocessing import ICA
from mne_icalabel import label_components

# Drop channels #10 and #21 (mastoids) before ICA
original_EEG_data_interp.drop_channels(['GND', 'Rmastoid'])

# Applying common average referencing (CAR) to comply with ICLabel requirements
# Not filtering because data has already been filtered above
filt_EEG = original_EEG_data_interp.set_eeg_reference("average")

# Define ICA parameters
n_components = 0.99 # Fitting as many components as are required to explain 99% variance, ignoring remaining dimensions
ica = ICA(
    n_components=n_components,
    max_iter="auto",
    method="infomax",
    random_state=97,
    fit_params=dict(extended=True),
)

# Fit ICA
ica.fit(original_EEG_data_interp, decim=3)
ica

# Plot the ICA components as time series
ica.plot_sources(filt_EEG, show_scrollbars=False, show=True)

# Plot the ICA components as topographies in multiple windows
print("Plotting the ICA components as topographies...")
n_components_actual = ica.n_components_
for i in range(0, n_components_actual, 62):
    ica.plot_components(picks=range(i, min(i + 62, n_components_actual)), ch_type='eeg', inst=filt_EEG)
    # blinks
    ica.plot_overlay(filt_EEG, exclude=[0], picks="eeg")

# Selecting ICA components automatically using ICLabel
ic_labels = label_components(filt_EEG, ica, method="iclabel")

# ICLabel scores
print(ic_labels["labels"])

# Extract labels and reconstruct raw data
labels = ic_labels["labels"]
exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
print(f"Excluding these ICA components: {exclude_idx}")

# Copy the original interpolated EEG data
reconst_EEG = filt_EEG.copy()

# Apply the ICA transformation, excluding certain components
ica.apply(reconst_EEG, exclude=exclude_idx)

# Plot the original data and set the window title
fig = original_EEG_data_interp.plot(show_scrollbars=False)
fig.set_window_title('Original EEG Data')

# Plot the reconstructed data and set the window title
fig = reconst_EEG.plot(show_scrollbars=False)
fig.set_window_title('Reconstructed EEG Data')

# Save the preprocessed data
output_dir = r'C:\Users\cerna\Downloads'
preprocessed_file = os.path.join(output_dir, subj + '_ICA.fif')
reconst_EEG.save(preprocessed_file, overwrite=True)

#################################################################################

# EPOCHING

# Define epoch parameters
name = subj + '_eventchan'  # --> change for each condition
epoch_no = np.floor(reconst_EEG.get_data().shape[1] / reconst_EEG.info['sfreq'])  # Latency rate/Sampling rate

# Create a list of onset times for your events
onsets = np.arange(0, reconst_EEG.get_data().shape[1] / reconst_EEG.info['sfreq'], 1)

# Create a list of event durations (all zeros if the events are instantaneous)
durations = np.zeros_like(onsets)

# Create a list of event descriptions
descriptions = ['Event'] * len(onsets)

# Create an Annotations object
annotations = mne.Annotations(onsets, durations, descriptions)

# Add the annotations to the Raw object
reconst_EEG.set_annotations(annotations)

# Now you can extract the events from the annotations
events, event_id = mne.events_from_annotations(reconst_EEG)

# Define epoching parameters
name = subj + '_epoch'  # --> change the name of condition
codes = ['1']
tmin = 0.0  # Start of the epoch (in seconds)
tmax = 60.0  # End of the epoch (in seconds)

# Create epochs without rejection to keep all data
epochs_all = mne.Epochs(reconst_EEG, events, event_id=event_id, tmin=tmin, tmax=tmax, proj=True, baseline=None, preload=True)

# Apply z-score normalization and keep track of which epochs exceed the threshold
zscore_threshold = 6
to_drop = []

for i in range(len(epochs_all)):
    epochs_all._data[i] = zscore(epochs_all._data[i], axis=1)
    if np.any(np.abs(epochs_all._data[i]) > zscore_threshold):
        to_drop.append(i)

# Now we can drop the epochs that exceeded the threshold
epochs_all.drop(to_drop)

# Resample and decimate the epochs
current_sfreq = epochs_all.info['sfreq']
desired_sfreq = 330  # Hz

# Apply the resampling
epochs_all.resample(desired_sfreq, npad='auto')

# Get the data from all epochs
data_all = epochs_all.get_data()

# Save the filtered data
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
preprocessed_file = os.path.join(output_dir, subj + '_epoched.fif')
epochs_all.save(preprocessed_file, overwrite=True)








