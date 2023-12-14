# Description: All-to-all connectivity at the level of networks

# Import necessary libraries
import os
import numpy as np
import mne
from nilearn import datasets
from scipy.signal import hilbert
import matplotlib.pyplot as plt
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout


# Set your output directory
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # Replace with your subject ID

# Load the label time courses
label_time_courses_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy")
label_time_courses = np.load(label_time_courses_file)

# Fetch the Schaefer atlas with 100 parcels
schaefer_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)

# Load labels from the atlas
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order',
                                    subjects_dir=r'C:\Users\cerna\mne_data\MNE-fsaverage-data')

# Group labels by network
networks = {}
for label in labels:
    # Extract network name from label name (assuming format: 'NetworkName_RegionName')
    network_name = label.name.split('_')[0]
    if network_name not in networks:
        networks[network_name] = []
    networks[network_name].append(label)

# Aggregate activity by network
network_time_courses = []
for network, network_labels in networks.items():
    # Find indices of labels belonging to this network
    indices = [labels.index(label) for label in network_labels]

    # Average activity across all regions in this network
    network_activity = np.mean(label_time_courses[:, indices, :], axis=1)
    network_time_courses.append(network_activity)

network_time_courses_np = np.array(network_time_courses)

# Compute dPLI
def compute_dPLI(data):
    n_regions = data.shape[0]
    dPLI_matrix = np.zeros((n_regions, n_regions))

    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)

    for i in range(n_regions):
        for j in range(n_regions):
            if i != j:
                phase_diff = phase_data[i] - phase_data[j]
                dPLI_matrix[i, j] = np.abs(np.mean(np.exp(complex(0, 1) * phase_diff)))

    return dPLI_matrix

# Compute dPLI for aggregated network activity
network_dPLI_matrix = compute_dPLI(network_time_courses_np)

# Get the network names
network_names = list(networks.keys())

# Create a circular layout for the networks
node_angles = circular_layout(network_names, network_names, start_pos=90)

# Plot the graph. We'll show the strongest connections, but you can adjust the number as needed.
fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
plot_connectivity_circle(network_dPLI_matrix, network_names, n_lines=300,
                         node_angles=node_angles, title='Network Connectivity using dPLI', ax=ax)
fig.tight_layout()
plt.show()
