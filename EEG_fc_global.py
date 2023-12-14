# Description: This script computes the functional connectivity of EEG data using the dPLI metric.

# Import packages
import os
import numpy as np
import mne
import networkx as nx
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Set your output directory
output_dir = r'C:\Users\cerna\Downloads'  # Replace with your desired output directory
subj = '101'  # Replace with your subject ID

# Load the label time courses
label_time_courses_file = os.path.join(output_dir, f"{subj}_label_time_courses.npy")
label_time_courses = np.load(label_time_courses_file)

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

# Sampling rate and window parameters
sampling_rate = 512  # in Hz
window_length_seconds = 1  # desired window length in seconds
step_size_seconds = 0.5  # desired step size in seconds

# Convert time to samples
window_length_samples = int(window_length_seconds * sampling_rate)  # convert to samples
step_size_samples = int(step_size_seconds * sampling_rate)  # convert to samples

# Calculate total duration in samples
num_epochs_per_hemisphere = label_time_courses.shape[0] / 2
duration_per_epoch = label_time_courses.shape[2] / sampling_rate
total_duration_samples = int(num_epochs_per_hemisphere * duration_per_epoch * sampling_rate)

# Time-resolved dPLI computation
num_windows = int((total_duration_samples - window_length_samples) / step_size_samples) + 1
windowed_dpli_matrices = []

# Compute dPLI for each window
for win_idx in range(num_windows):
    start_sample = win_idx * step_size_samples
    end_sample = start_sample + window_length_samples

    # Check if end_sample exceeds the total number of samples
    if end_sample > total_duration_samples:
        break

    # Calculate epoch and sample indices
    start_epoch = start_sample // label_time_courses.shape[2]
    start_sample_in_epoch = start_sample % label_time_courses.shape[2]
    end_epoch = end_sample // label_time_courses.shape[2]
    end_sample_in_epoch = end_sample % label_time_courses.shape[2]

    # Extract data across epochs
    if start_epoch == end_epoch:
        windowed_data = label_time_courses[start_epoch, :, start_sample_in_epoch:end_sample_in_epoch]
    else:
        first_part = label_time_courses[start_epoch, :, start_sample_in_epoch:]
        samples_needed_from_second_epoch = window_length_samples - first_part.shape[1]
        second_part = label_time_courses[end_epoch, :, :samples_needed_from_second_epoch]
        windowed_data = np.concatenate((first_part, second_part), axis=1)  # Change axis back to 1

    dpli_result = compute_dPLI(windowed_data)
    windowed_dpli_matrices.append(dpli_result)

# Check the number of windows in the list
num_of_windows = len(windowed_dpli_matrices)
print(f"Total number of windows: {num_of_windows}")

# Construct a graph
G = nx.from_numpy_array(dpli_result, create_using=nx.Graph)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def visualize_graph_colored_nodes(G):
    # Compute the degree of each node
    degrees = [G.degree(n) for n in G.nodes()]

    # Set up a layout for our nodes
    layout = nx.spring_layout(G)

    # Map the degrees to a color map (e.g., using a colormap like 'viridis')
    node_colors = plt.cm.viridis(degrees)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the nodes with the color map
    nx.draw_networkx_nodes(G, layout, node_size=500, node_color=node_colors, alpha=0.6, ax=ax)

    # Draw the edges
    nx.draw_networkx_edges(G, layout, width=1.0, alpha=0.5, ax=ax)

    # Draw the node labels
    nx.draw_networkx_labels(G, layout, font_size=10, ax=ax)

    # Add a colorbar to show the degree values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mcolors.Normalize(vmin=min(degrees), vmax=max(degrees)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Node Degree')

    ax.set_title("dPLI Graph Visualization with Colored Nodes")
    ax.axis("off")
    plt.show()

visualize_graph_colored_nodes(G)

#######################################################################################################################

# Graph theoretical computations

# 1. Check if the graph is connected
if not nx.is_connected(G):
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

# 2. Adjust the probability p for the Erdős–Rényi graph
p = len(G.edges()) / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2)
G_rand = nx.erdos_renyi_graph(G.number_of_nodes(), p=p)

# Compute graph theoretical metrics
avg_path_length = nx.average_shortest_path_length(G)
clustering_coefficient = nx.average_clustering(G)
modularity = nx.algorithms.community.modularity(G, nx.algorithms.community.greedy_modularity_communities(G))
local_efficiency = nx.local_efficiency(G)

# 3. Compute the average betweenness centrality
betweenness_dict = nx.betweenness_centrality(G)
avg_betweenness = sum(betweenness_dict.values()) / len(betweenness_dict)

global_efficiency = nx.global_efficiency(G)

# Compute Small-Worldness
C_rand = nx.average_clustering(G_rand)
L_rand = nx.average_shortest_path_length(G_rand)
small_worldness = (clustering_coefficient / C_rand) / (avg_path_length / L_rand)

# Display computed metrics
print(f"Time Window: 10")
print(f"Average Path Length: {avg_path_length}")
print(f"Average Clustering Coefficient: {clustering_coefficient}")
print(f"Modularity: {modularity}")
print(f"Local Efficiency: {local_efficiency}")
print(f"Average Betweenness Centrality: {avg_betweenness}")
print(f"Global Efficiency: {global_efficiency}")
print(f"Small-Worldness: {small_worldness}")
