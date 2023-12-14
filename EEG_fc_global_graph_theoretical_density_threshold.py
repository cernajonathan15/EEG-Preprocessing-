# Description: This script computes the functional connectivity of EEG data using the dPLI and PLI metrics

# Import packages
import os
import numpy as np
import mne
import networkx as nx
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


# Compute wPLI at the level of regions
def compute_wPLI(data):
    n_regions = data.shape[0]
    wPLI_matrix = np.zeros((n_regions, n_regions))

    # Compute the phase of the analytic signal
    analytic_signal = hilbert(data)
    phase_data = np.angle(analytic_signal)

    for i in range(n_regions):
        for j in range(i+1, n_regions):  # Only compute for upper triangle
            phase_diff = phase_data[i] - phase_data[j]
            imag_part = np.abs(np.imag(np.exp(1j * phase_diff)))
            wPLI_matrix[i, j] = np.mean(imag_part) / np.mean(np.abs(np.exp(1j * phase_diff)))
            wPLI_matrix[j, i] = wPLI_matrix[i, j]  # Symmetric matrix

    return wPLI_matrix


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
windowed_wpli_matrices = []

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

    # window-by-window dPLI computation
    dpli_result = compute_dPLI(windowed_data)
    wpli_result = compute_wPLI(windowed_data)

    # window-by-window wPLI computation
    windowed_dpli_matrices.append(dpli_result)
    windowed_wpli_matrices.append(wpli_result)

# Check the number of windows in the list
num_of_windows = len(windowed_dpli_matrices)
print(f"Total number of windows: {num_of_windows}")

# Construct Graphs
G_dPLI = nx.from_numpy_array(dpli_result, create_using=nx.DiGraph) #directed graph
G_wPLI = nx.from_numpy_array(wpli_result, create_using=nx.Graph) #undirected graph

# Apply a Density Threshold; Adjust density value as needed; means that we want to retain 10% of the strongest edges in the graph
def threshold_graph_by_density(G, density=0.1, directed=False):
    if density < 0 or density > 1:
        raise ValueError("Density value must be between 0 and 1.")

    num_edges_desired = int(G.number_of_edges() * density)
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    if directed:
        G_thresholded = nx.DiGraph()
    else:
        G_thresholded = nx.Graph()
    G_thresholded.add_edges_from(sorted_edges[:num_edges_desired])

    return G_thresholded

G_dPLI_thresholded = threshold_graph_by_density(G_dPLI, directed=True)
G_wPLI_thresholded = threshold_graph_by_density(G_wPLI, directed=False)

# Graph theoretical computations
# Check if the graph is connected
if not nx.is_weakly_connected(G_dPLI_thresholded):
    largest_cc = max(nx.weakly_connected_components(G_dPLI_thresholded), key=len)
    G_dPLI_thresholded = G_dPLI_thresholded.subgraph(largest_cc).copy()

if not nx.is_connected(G_wPLI_thresholded):
    largest_cc = max(nx.connected_components(G_wPLI_thresholded), key=len)
    G_wPLI_thresholded = G_wPLI_thresholded.subgraph(largest_cc).copy()

# Calculate edge density for dPLI and PLI thresholded graphs
p_dPLI = len(G_dPLI_thresholded.edges()) / (G_dPLI_thresholded.number_of_nodes() * (G_dPLI_thresholded.number_of_nodes() - 1))
p_wPLI = len(G_wPLI_thresholded.edges()) / (G_wPLI_thresholded.number_of_nodes() * (G_wPLI_thresholded.number_of_nodes() - 1))

# Compute graph theoretical metrics for dPLI
modularity_dPLI = nx.algorithms.community.modularity(G_dPLI_thresholded,
                                                     nx.algorithms.community.greedy_modularity_communities(
                                                         G_dPLI_thresholded))
clustering_coefficient_dPLI = nx.average_clustering(G_dPLI_thresholded)
avg_path_length_dPLI = nx.average_shortest_path_length(G_dPLI_thresholded)

# Convert directed graph to undirected for global efficiency calculation
G_dPLI_undirected = G_dPLI_thresholded.to_undirected()
global_efficiency_dPLI = nx.global_efficiency(G_dPLI_undirected)

betweenness_dict_dPLI = nx.betweenness_centrality(G_dPLI_thresholded)
avg_betweenness_dPLI = sum(betweenness_dict_dPLI.values()) / len(betweenness_dict_dPLI)

# Compute graph theoretical metrics for PLI
modularity_wPLI = nx.algorithms.community.modularity(G_wPLI_thresholded,
                                                    nx.algorithms.community.greedy_modularity_communities(
                                                        G_wPLI_thresholded))
clustering_coefficient_wPLI = nx.average_clustering(G_wPLI_thresholded)
avg_path_length_wPLI = nx.average_shortest_path_length(G_wPLI_thresholded)
global_efficiency_wPLI = nx.global_efficiency(G_wPLI_thresholded)
betweenness_dict_wPLI = nx.betweenness_centrality(G_wPLI_thresholded)
avg_betweenness_wPLI = sum(betweenness_dict_wPLI.values()) / len(betweenness_dict_wPLI)

# Small-worldness for dPLI
C_rand_dPLI = nx.average_clustering(nx.erdos_renyi_graph(G_dPLI_thresholded.number_of_nodes(), p_dPLI, directed=True))
L_rand_dPLI = nx.average_shortest_path_length(nx.erdos_renyi_graph(G_dPLI_thresholded.number_of_nodes(), p_dPLI, directed=True))
small_worldness_dPLI = (clustering_coefficient_dPLI / C_rand_dPLI) / (avg_path_length_dPLI / L_rand_dPLI)

# Small-worldness for PLI
C_rand_PLI = nx.average_clustering(nx.erdos_renyi_graph(G_wPLI_thresholded.number_of_nodes(), p_wPLI))
L_rand_PLI = nx.average_shortest_path_length(nx.erdos_renyi_graph(G_wPLI_thresholded.number_of_nodes(), p_wPLI))
small_worldness_wPLI = (clustering_coefficient_wPLI / C_rand_PLI) / (avg_path_length_wPLI / L_rand_PLI)

# Display computed metrics for dPLI
print(f"Metrics for dPLI:")
print(f"Modularity: {modularity_dPLI}")
print(f"Small-Worldness: {small_worldness_dPLI}")
print(f"Global Efficiency: {global_efficiency_dPLI}")
print(f"Average Clustering Coefficient: {clustering_coefficient_dPLI}")
print(f"Average Betweenness Centrality: {avg_betweenness_dPLI}")
print("\n")

# Display computed metrics for PLI
print(f"Metrics for PLI:")
print(f"Modularity: {modularity_wPLI}")
print(f"Small-Worldness: {small_worldness_wPLI}")
print(f"Global Efficiency: {global_efficiency_wPLI}")
print(f"Average Clustering Coefficient: {clustering_coefficient_wPLI}")
print(f"Average Betweenness Centrality: {avg_betweenness_wPLI}")

#######################################################################################################################

# Visualize the graph

# Import packages
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import matplotlib.colors as mcolors

# Define a function to visualize the graph

def visualize_graph_colored_nodes(G, title="Graph Visualization with Colored Nodes"):
    # Compute the degree of each node. For directed graphs, we'll use in_degree.
    if G.is_directed():
        degrees = [G.in_degree(n) for n in G.nodes()]
    else:
        degrees = [G.degree(n) for n in G.nodes()]

    # Set up a layout for our nodes
    layout = nx.spring_layout(G)

    # Map the degrees to a color map (e.g., using a colormap like 'viridis')
    node_colors = plt.cm.viridis(degrees)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the nodes with the color map
    nx.draw_networkx_nodes(G, layout, node_size=500, node_color=node_colors, alpha=0.6, ax=ax)

    # Draw the edges. For directed graphs, we'll use arrows.
    if G.is_directed():
        nx.draw_networkx_edges(G, layout, width=1.0, alpha=0.5, ax=ax, arrowstyle='-|>', arrowsize=20)
    else:
        nx.draw_networkx_edges(G, layout, width=1.0, alpha=0.5, ax=ax)

    # Draw the node labels
    nx.draw_networkx_labels(G, layout, font_size=10, ax=ax)

    # Add a colorbar to show the degree values
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=mcolors.Normalize(vmin=min(degrees), vmax=max(degrees)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Node Degree (In-degree for directed graphs)')

    ax.set_title(title)
    ax.axis("off")
    plt.show()

# Visualize the dPLI directed graph
visualize_graph_colored_nodes(G_dPLI_thresholded, title="dPLI Directed Graph Visualization with Colored Nodes")

# Visualize the PLI undirected graph
visualize_graph_colored_nodes(G_PLI_thresholded, title="PLI Undirected Graph Visualization with Colored Nodes")

#######################################################################################################################