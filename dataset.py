import numpy as np
from scipy.spatial.distance import pdist, squareform
import torch

def load_LKH_predictions(tour_filename):
    with open(tour_filename, "r") as tour_f:
        tour_lines = tour_f.readlines()

    #edges_nb = tour_lines[0].count(" ")
    #print(tour_lines[0].replace(" \n", "").replace("\n", "").split(" "))
    edges_nb = max([int(vertex_str) for vertex_str in tour_lines[0].replace(" \n", "").replace("\n", "").split(" ")])
    graphs_nb = len(tour_lines)

    # Numpy array keeping in the edges used for tours in a matrix
    tour_array = np.zeros((graphs_nb, edges_nb, edges_nb))

    for graph_idx in range(len(tour_lines)):
        line = tour_lines[graph_idx].replace(" \n", "").replace("\n", "") + " 1"
        #print(line, line.split(" "))
        line = [int(vertex_str)-1 for vertex_str in line.split(" ")]
        for vertex_idx in range(len(line)-1):
            tour_array[graph_idx, line[vertex_idx], line[vertex_idx+1]] = 1
            #tour_array[graph_idx, line[vertex_idx+1], line[vertex_idx]] = 1

    # Creating the tensor and the variable
    tour_tensor = torch.Tensor(tour_array)
    y_preds = torch.autograd.Variable(tour_tensor)
    return y_preds

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
        self.current_graph_idx += 1

        # Retrieve the optimal solution
        # Numpy array keeping in the edges used for tours in a matrix
        tour_array = np.zeros((self.num_graphs, self.num_nodes, self.num_nodes))

        for graph_idx in range(len(self.solution_data)):
            line = self.solution_data[graph_idx].replace(" \n", "").replace("\n", "") + " 1"
            line = [int(vertex_str)-1 for vertex_str in line.split(" ")]
            for vertex_idx in range(len(line)-1):
                tour_array[graph_idx, line[vertex_idx], line[vertex_idx+1]] = 1

        # Creating the tensor and the variable
        tour_tensor = torch.Tensor(tour_array)
        y_preds = torch.autograd.Variable(tour_tensor)


        return cost_array, y_preds, coords_array