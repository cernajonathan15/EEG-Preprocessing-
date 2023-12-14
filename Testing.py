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

import matplotlib
import mne
import numpy as np
from scipy.stats import zscore

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

# Create a dictionary for channel types
channel_types = {}

# Set all channels to 'eeg' by default
for ch in ch_map.values():
    channel_types[ch] = 'eeg'

# Update the dictionary with the special channel types
channel_types['RVEye'] = 'eog'
channel_types['LHEye'] = 'eog'
channel_types['RHEye'] = 'eog'
channel_types['Rneck'] = 'emg'
channel_types['Lneck'] = 'emg'
channel_types['Rmastoid'] = 'misc'
channel_types['GND'] = 'misc'

# Set the channel types in the EEG data
EEG.set_channel_types(channel_types)

# Apply a low-pass filter with a cutoff of 100 Hz
EEG.filter(l_freq=None, h_freq=100)

# Apply a high-pass filter with a cutoff of 0.5 Hz
EEG.filter(l_freq=0.5, h_freq=None)

# Add a notch filter from 60 Hz
freqs = np.arange(60, 241, 60)  # This will create an array [60, 120, 180, 240] to capture the harmonics
EEG.notch_filter(freqs)

# Plot the data to visualize waveforms after filtering
EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# Plotting EEG signal via PSD to check if the notch filter removed the power line noise
EEG.plot_psd()

# Save the filtered data
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
preprocessed_file = os.path.join(output_dir, subj + '_maprenamed&nfiltered.fif')
EEG.save(preprocessed_file, overwrite=True)

#################################################################################
# OLD PROCEDURE: Re-referencing to average of mastoids

# Plot the data to visualize waveforms before re-referencing
#EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# Re-referencing to average of mastoids
#these two reference channels will be automatically dropped out for ICA
#EEG.set_eeg_reference(ref_channels=['Ch10', 'Ch21'])  #Green 10 = Left mastoid; Green 21 = Right mastoid

# Plot the data to visualize waveforms after re-referencing
#EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# Plotting EEG signal via PSD to identify potential noise that needs to filtered
#EEG.plot_psd()

# NEW PROCEDURE: Reference-free CSD (current source density) transformation

# Plot the data to visualize waveforms before CSD
EEG.plot(n_channels=len(EEG.ch_names), scalings='auto')

# Check for problematic channels
problematic_channels = []

for ch in EEG.info['chs']:
    x, y, z = ch['loc'][:3]
    if np.isnan(x) or np.isnan(y) or np.isnan(z) or x == 0 or y == 0 or z == 0:
        problematic_channels.append(ch['ch_name'])

print("Problematic channels:", problematic_channels)

# Drop the problematic channels
EEG.drop_channels(problematic_channels)

# Apply CSD transformation
EEG_csd = mne.preprocessing.compute_current_source_density(inst=EEG, stiffness=4, lambda2=1e-05, n_legendre_terms=50, copy=True, verbose=None)

# Plot the data to visualize waveforms after CSD
EEG_csd.plot(n_channels=len(EEG_csd.ch_names), scalings='auto')

# Create a dictionary mapping from 'csd' channel names to 'eeg'
ch_type_dict = {ch_name: 'eeg' for ch_name in EEG_csd.ch_names if EEG_csd.get_channel_types(picks=[ch_name])[0] == 'csd'}

# Change the unit of the channels you're trying to set as EEG
for ch_name in ch_type_dict.keys():
    idx = EEG_csd.ch_names.index(ch_name)
    EEG_csd.info['chs'][idx]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_V

# Set the channel types
EEG_csd.set_channel_types(ch_type_dict)

# Verify the changes
print(EEG_csd.get_channel_types())

# Save the CSD transformation
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
preprocessed_file = os.path.join(output_dir, subj + '_csd.fif')
EEG_csd.save(preprocessed_file, overwrite=True)

#################################################################################

# MARKING BAD CHANNELS

import copy # This is a Python module that allows you to copy objects without changing the original object

# Visualize all channels to identify bad channels
EEG_csd.plot(n_channels=len(EEG.ch_names), scalings='auto')

# This can be used to plot the data with the bad channels marked.
# Uncomment the two lines of code below to see the plot
picks = mne.pick_channels_regexp(EEG_csd.ch_names, regexp="AF.|FT.")  # Replace 'regexp=" ."' with the tentative bad channels
EEG_csd.plot(order=picks, n_channels=len(picks))

# Change list of bad channels
original_bads = copy.deepcopy(EEG_csd.info["bads"])
EEG_csd.info["bads"].append("AF7")  # add a single channel
original_bads.append("AF7")  # add a single channel to the original_bads list
#EEG_csd.info["bads"].extend(["EEG 051", "EEG 052"])  # add a list of channels
#original_bads["bads"].extend(["EEG 051", "EEG 052"])  # add a list of channels

# Print the bad channels to double check
print(EEG.info['bads'])
print(original_bads)

# Save the data with the bad channels marked
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
preprocessed_file = os.path.join(output_dir, subj + '_badchannels.fif')
EEG.save(preprocessed_file, overwrite=True)

#############################-ONLY RUN NEXT STEP IF BAD CHANNELS ARE FOUND TO INTERPOLATE-#############################

# ICA (Independent Component Analysis)

from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.decomposition import PCA

# Keep a reference to the original, uncropped data
original_EEG = EEG

# Crop a copy of the data to three seconds for easier plotting
cropped_EEG = EEG.copy().crop(tmin=0, tmax=3).load_data()

# Drop channels #10 and #21 (mastoids) before ICA
original_EEG.drop_channels(['GND']) # Include Rmastoid if not dropped above for CSD

# Determine the number of PCA components
data = original_EEG.get_data().T
pca = PCA()
pca.fit(data)

# Plot the explained variance ratio
plt.figure(figsize=(10, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()

# Define ICA parameters
n_components = 0.99  # Use the number of PCA components for ICA
ica = ICA(
    n_components=n_components,
    max_iter="auto",
    method="infomax",
    random_state=97,
    fit_params=dict(extended=True),
)

# Pick only EEG channels
picks_eeg = mne.pick_types(original_EEG.info, meg=False, eeg=True, eog=False, stim=False, emg=False, exclude='bads')

# Fit ICA using only EEG channels
ica.fit(original_EEG, picks=picks_eeg, decim=3)

# Plot the ICA components as time series
ica.plot_sources(original_EEG, show_scrollbars=False, show=True)

# Plot the ICA components as topographies in multiple windows
print("Plotting the ICA components as topographies...")
n_components_actual = ica.n_components_

# Selecting ICA components automatically using ICLabel
ic_labels = label_components(original_EEG, ica, method="iclabel")
component_labels = ic_labels["labels"]  # Extract the labels

for i in range(n_components_actual):
    # Plot the component
    fig = ica.plot_components(picks=[i], ch_type='eeg', inst=original_EEG)

    # Display the current label and ask if the user wants to change it
    print(f"Current label for component {i}: {component_labels[i]}")
    change = input("Do you want to change this label? (yes/no): ").strip().lower()

    if change == 'yes':
        new_label = input("Enter the new label: ").strip()
        component_labels[i] = new_label
        fig.axes[0].set_title(new_label)
    else:
        fig.axes[0].set_title(component_labels[i])

    # Close the figure to move to the next component
    plt.close(fig)

# ICLabel scores
print(component_labels)

# Extract labels and reconstruct raw data --> need to see if there is a way to set an .80 threshold for brain and other components
labels = ic_labels["labels"]
exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
print(f"Excluding these ICA components: {exclude_idx}")

# Copy the original interpolated EEG data
reconst_EEG = original_EEG.copy()

# Apply the ICA transformation, excluding certain components
ica.apply(reconst_EEG, exclude=exclude_idx)

# Plot the original data and set the window title
fig = original_EEG.plot(show_scrollbars=False)
fig.set_window_title('Original EEG Data')

# Plot the reconstructed data and set the window title
fig = reconst_EEG.plot(show_scrollbars=False)
fig.set_window_title('Reconstructed EEG Data')

# Save the preprocessed data
output_dir = r'C:\Users\cerna\Downloads'
preprocessed_file = os.path.join(output_dir, subj + '_ICA.fif')
reconst_EEG.save(preprocessed_file, overwrite=True)

#################################################################################

# Interpolate bad channels

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
tmin = -0.5  # Start of the epoch (in seconds)
tmax = 0.5  # End of the epoch (in seconds)

# Create epochs without rejection to keep all data
epochs_all = mne.Epochs(reconst_EEG, events, event_id=event_id, tmin=tmin, tmax=tmax, proj=True, baseline=None, preload=True)

# Apply z-score normalization and keep track of which epochs exceed the threshold
zscore_threshold = 6
to_drop = []

temp_data = np.zeros_like(epochs_all._data)

for i in range(len(epochs_all)):
    temp_data[i] = zscore(epochs_all._data[i], axis=1)
    if np.any(np.abs(temp_data[i]) > zscore_threshold):
        to_drop.append(i)

# Now we can drop the epochs that exceeded the threshold
epochs_all.drop(to_drop)

# Resample and decimate the epochs
current_sfreq = epochs_all.info['sfreq']
desired_sfreq = 512  # Hz chaging this according to  https://doi.org/10.1046/j.1440-1819.2000.00729.x

# Apply the resampling
epochs_all.resample(desired_sfreq, npad='auto')

# Get the data from all epochs
data_all = epochs_all.get_data()

# Plot the data
epochs_all.plot()

# Save the filtered data
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
preprocessed_file = os.path.join(output_dir, subj + '_epoched.fif')
epochs_all.save(preprocessed_file, overwrite=True)








