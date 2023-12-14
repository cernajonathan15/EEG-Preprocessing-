
# Intended to be used after the EEG_source_to_parcel_&_all_to_all_connectivity.py script

# For visualization, select one of the windows... or not
#chosen_window = 3
#dPLI_matrix = windowed_dpli_matrices[chosen_window] # can delete [chosen_window] to get the entire matrix

#G_dPLI = nx.convert_matrix.from_numpy_array(dPLI_matrix)
#G_dPLI_thresholded = disparity_filter(G_dPLI)

# Visualization using dPLI
node_angles = np.linspace(0, 360, len(ordered_regions), endpoint=False)
dPLI_matrix_thresholded = np.zeros_like(dPLI_matrix) # Initialize a matrix of zeros with the same shape as dPLI_matrix
for i, j, data in G_dPLI_thresholded.edges(data=True): # Iterate through the edges of the thresholded graph
    dPLI_matrix_thresholded[i, j] = data['weight']
    dPLI_matrix_thresholded[j, i] = data['weight']

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', subplot_kw=dict(polar=True))
plot_connectivity_circle(dPLI_matrix_thresholded, ordered_regions, n_lines=300, node_angles=node_angles,
                         title='Thresholded Regional Connectivity using dPLI', ax=ax)
fig.tight_layout()
plt.show()



# Dynamics representation (animation) across all windows from circular graph
from matplotlib.animation import FuncAnimation

node_angles = np.linspace(0, 360, len(ordered_regions), endpoint=False)


def threshold_matrix(matrix):
    G_temp = nx.convert_matrix.from_numpy_array(matrix)
    G_temp_thresholded = disparity_filter(G_temp)

    matrix_thresholded = np.zeros_like(matrix)
    for i, j, data in G_temp_thresholded.edges(data=True):
        matrix_thresholded[i, j] = data['weight']
        matrix_thresholded[j, i] = data['weight']
    return matrix_thresholded


fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', subplot_kw=dict(polar=True))
plot_connectivity_circle(threshold_matrix(windowed_dpli_matrices[0]), ordered_regions, n_lines=300,
                         node_angles=node_angles,
                         title='Thresholded Regional Connectivity using dPLI for Window 0', ax=ax)


def update(window_number):
    ax.clear()
    current_matrix = threshold_matrix(windowed_dpli_matrices[window_number])
    plot_connectivity_circle(current_matrix, ordered_regions, n_lines=300, node_angles=node_angles,
                             title=f'Thresholded Regional Connectivity using dPLI for Window {window_number}', ax=ax)
    return ax,


ani = FuncAnimation(fig, update, frames=num_of_windows, blit=True)
plt.show()


# Histogram of Connectivity Values for the chosen window using dPLI
plt.hist(dPLI_matrix.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Distribution of dPLI Connectivity Values')
plt.xlabel('Connectivity Value')
plt.ylabel('Frequency')
plt.show()

# Histogram of Connectivity Values for the chosen window using Cross-Correlation
plt.hist(CrossCorr_matrix.flatten(), bins=50, color='red', alpha=0.7)
plt.title('Distribution of Cross-Correlation Connectivity Values')
plt.xlabel('Connectivity Value')
plt.ylabel('Frequency')
plt.show()