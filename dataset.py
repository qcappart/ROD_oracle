import numpy as np
from scipy.spatial.distance import pdist, squareform

class DataReader():
    """Class that reads and treats TSP data
    """
    def __init__(self, num_nodes, file_path, solution_path):
        self.num_nodes = num_nodes
        self.file_path = file_path
        self.file_data = open(file_path, "r").readlines()
        self.num_graphs = len(self.file_data)
        self.current_graph_idx = 0
        if solution_path is None:
            # Not implemented for now
            print("Come back later")
            self.no_solution = True
        else:
            self.solution_data = open(solution_path, "r").readlines()
            self.no_solution = False

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

        # Retrieve the optimal solution
        forward_opt_tour = {}
        backward_opt_tour = {}

        line = self.solution_data[self.current_graph_idx].replace(" \n", "").replace("\n", "") + " 1"
        line = [int(vertex_str)-1 for vertex_str in line.split(" ")]
        for vertex_idx in range(len(line)-1):
            forward_opt_tour[line[vertex_idx]] = line[vertex_idx+1]
            backward_opt_tour[line[vertex_idx+1]] = line[vertex_idx]

        self.current_graph_idx += 1
        return cost_array, coords_array, forward_opt_tour, backward_opt_tour