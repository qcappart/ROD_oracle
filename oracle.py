import os
import json
import argparse
import time

import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import DataReader
from concorde_translater import create_concorde_input, call_concorde_solver, retrieve_concorde_output


def compute_tour_length(tour_nodes, cost_array):
    """Compute a tour length given the cost array
    """
    total_length = 0
    nodes_nb, _ = cost_array.shape
    for node_idx in range(nodes_nb-1):
        total_length += cost_array[tour_nodes[node_idx], \
                                    tour_nodes[node_idx+1]]
    # Closing the tour
    total_length += cost_array[tour_nodes[0], \
                                tour_nodes[nodes_nb-1]]
    return total_length


def display_current_situation(cost_array, node_coords, visited_vertices, num_nodes, forward_opt_tour):
    """Display with matplotlib the evolution of the tour construction
    """
    # Determining if the tour is being followed forwards or backwards
    if (forward_opt_tour[visited_vertices[0]]==visited_vertices[-1]):
        # Tour followed forwards
        current_vertex = visited_vertices[-1]
        final_vertex = visited_vertices[0]
    else:
        # Tour followed backwards
        current_vertex = visited_vertices[0]
        final_vertex = visited_vertices[-1]
    # Reconstruct the partial tour
    opt_tour_vertices = [current_vertex]
    begin_opt = True
    while (current_vertex != final_vertex or begin_opt):
        begin_opt = False
        next_opt_vertex = forward_opt_tour[current_vertex]
        opt_tour_vertices.append(next_opt_vertex)
        current_vertex = next_opt_vertex

    # Compute the length of the optimal tour including the already constructed partial tour
    current_length = 0
    for idx in range(len(opt_tour_vertices)-1):
        current_length += cost_array[opt_tour_vertices[idx], opt_tour_vertices[idx+1]]
    for idx in range(len(visited_vertices)-1):
        current_length += cost_array[visited_vertices[idx], visited_vertices[idx+1]]

    # Plot the current partial tour in blue
    plt.plot(node_coords[:, 0], node_coords[:, 1], "bo")
    plt.plot(node_coords[visited_vertices, 0], node_coords[visited_vertices, 1], "b-")
    # Plot the optimal halmitonian path closing the min length tour in red
    plt.plot(node_coords[opt_tour_vertices, 0], node_coords[opt_tour_vertices, 1], "r-")
    plt.title("Current optimal distance : " + str(current_length))
    plt.pause(0.5)
    plt.clf()


def compute_tour_nodes(oracle_precision, args, cost_array, coords_array, forward_opt_tour, backward_opt_tour):
    num_nodes = cost_array.shape[0]
    current_vertex = 0
    visited_vertices_nb = 1
    visited_vertices = [0]
    nodes_list = range(num_nodes)
    for visited_vertices_nb in range(1, num_nodes):
        if (np.random.rand()<=oracle_precision):
            # Do the optimal choice
            # Looking at the vertices before and after and choosing the one not visited yet
            new_vertex_1 = forward_opt_tour[current_vertex]
            new_vertex_2 = backward_opt_tour[current_vertex]
            if new_vertex_1 not in visited_vertices:
                new_vertex = new_vertex_1
            else :
                new_vertex = new_vertex_2
        else:
            # Make a random guess
            # Compute the probability distribution to choose the new vertex
            vertex_scores = cost_array[current_vertex, :]
            # Put a dummy value so that python does not complain about the division by zero
            # This vertex is marked as visited so it will not affect the random choice
            vertex_scores[current_vertex] = 1
            distances = 1.0/vertex_scores
            for visited_idx in visited_vertices:
                distances[visited_idx] = 0
            distances = distances/distances.sum()
            new_vertex = np.random.choice(range(num_nodes), p=distances)

        # Add the new vertex to the partial tour and check if it was the optimal one
        visited_vertices.append(new_vertex)
        chose_optimal_arc = (forward_opt_tour[current_vertex]==new_vertex) or (backward_opt_tour[current_vertex]==new_vertex)

        if not chose_optimal_arc:
            # Figure out the nodes ordering in order to reconstruct the optimal tour after the Concorde computation
            partial_nodes_list = [node_idx for node_idx in nodes_list if node_idx not in visited_vertices[1:-1]]
            invert_partial_list = dict((val, key) for key,val in enumerate(partial_nodes_list))
            partial_array = np.array(partial_nodes_list)

            # Construct the cost array to find the optimal halmitonian path closing the min length tour
            restricted_cost_array = np.delete(np.delete(cost_array, visited_vertices[1:-1], axis=0), visited_vertices[1:-1], axis=1)
            new_tsp_array = np.zeros((restricted_cost_array.shape[0]+1, restricted_cost_array.shape[1]+1))
            # Copy the distances from the original cost array
            new_tsp_array[:-1, :-1] = restricted_cost_array[:, :]
            # Put upper bound values on the connections to the dummy vertex
            upper_bound = num_nodes*restricted_cost_array.max()
            new_tsp_array[-1, 1:] = upper_bound
            new_tsp_array[1:, -1] = upper_bound
            # Put distances to zero for the two vertices supposed to be linked to the dummy
            new_tsp_array[-1, -1] = 0            
            new_tsp_array[invert_partial_list[visited_vertices[-1]], -1] = 0
            new_tsp_array[-1, invert_partial_list[visited_vertices[-1]]] = 0
            
            # Call Concorde for the optimization part
            create_concorde_input(args.temp_file_path, new_tsp_array*args.cost_multiplier)
            solution_path = call_concorde_solver(args.temp_file_path)
            new_tsp_tour = retrieve_concorde_output(solution_path)
            dummy_vertex_idx = new_tsp_array.shape[0]-1
            new_tsp_tour = new_tsp_tour[new_tsp_tour!=dummy_vertex_idx]
            new_tsp_tour = np.append(new_tsp_tour, new_tsp_tour[0])
            
            # Update the optimal solution
            forward_opt_tour = {}
            backward_opt_tour = {}
            for tour_idx in range(len(new_tsp_tour)-1):
                forward_opt_tour[partial_array[new_tsp_tour][tour_idx]] = partial_array[new_tsp_tour][tour_idx+1]
                backward_opt_tour[partial_array[new_tsp_tour][tour_idx+1]] = partial_array[new_tsp_tour][tour_idx]

        elif chose_optimal_arc and (current_vertex!=visited_vertices[0]):
            # Remove the current vertex from the hamiltonian path
            pred_vertex = backward_opt_tour[current_vertex]
            next_vertex = forward_opt_tour[current_vertex] 
            del forward_opt_tour[current_vertex]
            del backward_opt_tour[current_vertex]
            forward_opt_tour[pred_vertex] = next_vertex
            backward_opt_tour[next_vertex] = pred_vertex

        # Else case : the current vertex is the start vertex
        # The first edge has been selected and is optimal
        # There is no need to upgrade forward_opt_tour and backward_opt_tour

        current_vertex = new_vertex
        # Show the current state of the oracle construction
        if args.display:
            display_current_situation(cost_array, coords_array, visited_vertices, num_nodes, forward_opt_tour)

    return np.array(visited_vertices)


if __name__=="__main__":
    # Get the command information
    parser = argparse.ArgumentParser(description='Parser for the oracle algorithm')
    parser.add_argument('--data', type=str, required=True, \
                        help='Path to the instance file')
    parser.add_argument('--display', action='store_true', \
                        help='Use this flag to see the construction')
    parser.add_argument('--min_prec', type=float, default=0, \
                        help='Min precision for the oracle')
    parser.add_argument('--max_prec', type=float, default=1, \
                        help='Max precision for the oracle')
    parser.add_argument('--prec_step', type=float, default=0.01, \
                        help='Step precision for the precision increase')
    parser.add_argument('--cost_multiplier', type=int, default=5*1e2, \
                        help='Multiplier for the values of the cost array')
    parser.add_argument('--temp_file_path', type=str, \
                        default='temp_instance.tsp', \
                        help='Path to the temporary file created to call Concorde')
    args = parser.parse_args()

    # Prepare the precision values for the oracle
    precision_values = np.arange(args.min_prec, args.max_prec, args.prec_step)

    for oracle_precision in precision_values:
        print("Exploring with precision : " + str(oracle_precision))
        dataset = DataReader(args.data)
        opt_gap_sum = 0
        for graph_idx in tqdm(range(dataset.num_graphs)):
            # Retrieve the graph information
            cost_array, coords_array = dataset.get_next_graph()

            # Constructing the optimal solution for this instance
            create_concorde_input(args.temp_file_path, args.cost_multiplier*cost_array)
            solution_path = call_concorde_solver(args.temp_file_path)
            optimal_tour = retrieve_concorde_output(solution_path)
            tsp_tour = np.append(optimal_tour, optimal_tour[0])
            forward_opt_tour = {}
            backward_opt_tour = {}
            for tour_idx in range(len(tsp_tour)-1):
                forward_opt_tour[tsp_tour[tour_idx]] = tsp_tour[tour_idx+1]
                backward_opt_tour[tsp_tour[tour_idx+1]] = tsp_tour[tour_idx]
            opt_length = compute_tour_length(optimal_tour, cost_array)


            # Computing a tour based on the oracle
            tour_nodes = compute_tour_nodes(oracle_precision, args, \
                                            cost_array, coords_array, \
                                            forward_opt_tour, backward_opt_tour)
            tour_length = compute_tour_length(tour_nodes, cost_array)
            optimality_gap = (tour_length-opt_length)/opt_length
            opt_gap_sum += optimality_gap

        print("Mean tour length : " + str(opt_gap_sum/dataset.num_graphs))