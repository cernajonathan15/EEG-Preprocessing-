
# This script is used to build the a two-step thresholding procedure (bootstrapping + disparity filter)
# The output should be a list of thresholded dPLI matrices for each window per participant

#import libraries
import numpy as np
import networkx as nx
from scipy.signal import hilbert
import mne
import os

# Set your output directory
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # Replace so that it loops over all participants

# Load the label time courses
label_time_courses_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy") # Make sure it loops over all participants
label_time_courses = np.load(label_time_courses_file)

# Load labels from the atlas
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', subjects_dir=r'C:\Users\cerna\mne_data\MNE-fsaverage-data')

# Function to compute dPLI for a given data window
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

# Function to perform aggregated bootstrapping and find optimal alpha and upper threshold
def aggregated_bootstrapping_and_threshold(windowed_graphs, num_iterations=1000, percentile=95, alpha_start=0.001,
                                           alpha_end=0.1, num_alphas=100):
    # Aggregate edge weights from all windowed graphs
    all_edge_weights = np.concatenate(
        [np.array([data['weight'] for _, _, data in G.edges(data=True)]) for G in windowed_graphs])

    # Perform bootstrapping on aggregated edge weights
    bootstrap_weights = []
    for _ in range(num_iterations):
        random_weights = np.random.choice(all_edge_weights, size=len(all_edge_weights), replace=True)
        bootstrap_weights.extend(random_weights)

    # Determine upper threshold for aggregated data
    upper_threshold = np.percentile(bootstrap_weights, percentile)

    # Test range of alphas to determine optimal alpha for aggregated data
    alphas = np.linspace(alpha_start, alpha_end, num_alphas)
    avg_connectivities = []
    for alpha in alphas:
        connectivities = []
        for G in windowed_graphs:
            G_filtered = G.copy()
            for u, v, weight in G.edges(data='weight'):
                if weight > upper_threshold or (G_filtered[u][v]['weight'] ** 2 / sum(
                        [d['weight'] ** 2 for _, _, d in G_filtered.edges(u, data=True)]) < alpha):
                    G_filtered.remove_edge(u, v)
            connectivities.append(np.mean(
                nx.convert_matrix.to_numpy_array(G_filtered)[np.nonzero(nx.convert_matrix.to_numpy_array(G_filtered))]))
        avg_connectivities.append(np.mean(connectivities))

    optimal_alpha_idx = np.argmin(np.abs(np.diff(avg_connectivities)))
    return alphas[optimal_alpha_idx], upper_threshold


# Function to apply aggregated threshold and disparity filter to a graph
def apply_aggregated_filter(G, optimal_alpha, upper_threshold):
    G_filtered = G.copy()
    for u, v, data in G.edges(data=True):
        if data['weight'] > upper_threshold:
            G_filtered.remove_edge(u, v)

        elif data['weight'] ** 2 / sum(
                [d['weight'] ** 2 for _, _, d in G_filtered.edges(u, data=True)]) < optimal_alpha:
            G_filtered.remove_edge(u, v)

    return G_filtered

# Sampling rate and window parameters
sampling_rate = 512  # in Hz
window_length_seconds = 1  # desired window length in seconds
step_size_seconds = 0.5  # desired step size in seconds

# Convert time to samples
window_length_samples = int(window_length_seconds * sampling_rate)
step_size_samples = int(step_size_seconds * sampling_rate)

# Calculate total duration in samples
num_epochs_per_hemisphere = label_time_courses.shape[0] / 2
duration_per_epoch = label_time_courses.shape[2] / sampling_rate
total_duration_samples = int(num_epochs_per_hemisphere * duration_per_epoch * sampling_rate)

# Compute dPLI for each window
num_windows = int((total_duration_samples - window_length_samples) / step_size_samples) + 1
windowed_dpli_matrices = []

for win_idx in range(num_windows):
    start_sample = win_idx * step_size_samples
    end_sample = start_sample + window_length_samples

    if end_sample > total_duration_samples:
        break

    start_epoch = start_sample // label_time_courses.shape[2]
    start_sample_in_epoch = start_sample % label_time_courses.shape[2]
    end_epoch = end_sample // label_time_courses.shape[2]
    end_sample_in_epoch = end_sample % label_time_courses.shape[2]

    if start_epoch == end_epoch:
        windowed_data = label_time_courses[start_epoch, :, start_sample_in_epoch:end_sample_in_epoch]
    else:
        first_part = label_time_courses[start_epoch, :, start_sample_in_epoch:]
        samples_needed_from_second_epoch = window_length_samples - first_part.shape[1]
        second_part = label_time_courses[end_epoch, :, :samples_needed_from_second_epoch]
        windowed_data = np.concatenate((first_part, second_part), axis=1)

    dpli_result = compute_dPLI(windowed_data)
    windowed_dpli_matrices.append(dpli_result)

# Convert each windowed dPLI matrix to a graph
windowed_graphs = [nx.convert_matrix.from_numpy_array(matrix, create_using=nx.DiGraph) for matrix in windowed_dpli_matrices]

# Perform aggregated bootstrapping and find optimal alpha and upper threshold
optimal_alpha, upper_threshold = aggregated_bootstrapping_and_threshold(windowed_graphs, num_iterations=1000, percentile=95)

# Apply the aggregated filter to each windowed graph
thresholded_dpli_matrices = []
for G_dPLI in windowed_graphs:
    G_dPLI_thresholded = apply_aggregated_filter(G_dPLI, optimal_alpha, upper_threshold)
    dpli_matrix_thresholded = nx.convert_matrix.to_numpy_array(G_dPLI_thresholded)
    thresholded_dpli_matrices.append(dpli_matrix_thresholded)

# thresholded_dpli_matrices contains the thresholded dPLI matrices for each window
