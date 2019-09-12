import numpy as np
from scipy.spatial.distance import pdist, squareform

class DataReader():
    """Class that reads and treats TSP data
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_data = open(file_path, "r").readlines()
        # Retrieve the number of nodes in the graphs
        first_data_line = self.file_data[0].split(" ")
        first_line_coords = [float(coord.replace("\n", "")) for coord in first_data_line]
        self.num_nodes = len(first_line_coords)//2
        self.num_graphs = len(self.file_data)
        self.current_graph_idx = 0

    def get_next_graph(self):
        # Coordinates follow this pattern : [x1, y1, x2, y2, ..., xn, yn]
        data_line = self.file_data[self.current_graph_idx].split(" ")
        data_coords = [float(coord.replace("\n", "")) for coord in data_line]
        # Put the coordinates in an array
        coords_array = np.zeros((self.num_nodes, 2))
        coords_array[:, 0] = data_coords[0::2]
        coords_array[:, 1] = data_coords[1::2]
        # Compute the pairwise distances
        cost_array = squareform(pdist(coords_array, 'euclidean'))

        self.current_graph_idx += 1
        return cost_array, coords_array