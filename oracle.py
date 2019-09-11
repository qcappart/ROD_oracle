import os
import json
import argparse
import time

import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from concorde.tsp import TSPSolver
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from dataset import DataReader
from concorde_translater import create_concorde_input, call_concorde_solver, retrieve_concorde_output

parser = argparse.ArgumentParser(description='gcn_tsp_parser')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--solution', type=str, required=True)
parser.add_argument('--display', action='store_true')

args = parser.parse_args()

def compute_mean_tour_length(num_graphs, bs_nodes, x_edges_values):
    total_length = 0
    nodes_nb, _ = x_edges_values.shape
    for node_idx in range(nodes_nb-1):
        total_length += x_edges_values[bs_nodes[node_idx], \
                                       bs_nodes[node_idx+1]]
    total_length += x_edges_values[bs_nodes[0], \
                                   bs_nodes[nodes_nb-1]]
    return total_length


def display_current_situation(cost_array, node_coords, visited_vertices, num_nodes, forward_opt_tour):
    # Determining if the tour is being followed forwards or backwards
    if (forward_opt_tour[visited_vertices[0]]==visited_vertices[-1]):
        # Tour followed forwards
        current_vertex = visited_vertices[-1]
        final_vertex = visited_vertices[0]
    else:
        # Tour followed backwards
        current_vertex = visited_vertices[0]
        final_vertex = visited_vertices[-1]
    print("current", visited_vertices, current_vertex)
    print("opt tour", forward_opt_tour, current_vertex)
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

    print(visited_vertices, opt_tour_vertices)

    # Plot the current partial tour
    plt.plot(node_coords[:, 0], node_coords[:, 1], "bo")
    plt.plot(node_coords[visited_vertices, 0], node_coords[visited_vertices, 1], "b-")
    # Plot the optimal halmitonian path closing the min length tour enclosing 
    plt.plot(node_coords[opt_tour_vertices, 0], node_coords[opt_tour_vertices, 1], "r-")
    plt.title("Current optimal distance : " + str(current_length))
    plt.pause(0.5)
    # plt.show()
    plt.clf()


def compute_tour_nodes(cost_array, num_nodes, oracle_precision, coords_array, forward_opt_tour, backward_opt_tour):
    chose_optimal_arc = True
    current_vertex = 0
    visited_vertices_nb = 1
    visited_vertices = [0]
    nodes_list = range(num_nodes)
    for visited_vertices_nb in range(1, num_nodes):
        problem = False

        if (np.random.rand()<=oracle_precision):
            print("GOOD")
            # Do the optimal choice
            # Looking at the vertices before and after and choosing the one not visited yet
            new_vertex_1 = forward_opt_tour[current_vertex]
            new_vertex_2 = backward_opt_tour[current_vertex]
            if new_vertex_1 not in visited_vertices:
                new_vertex = new_vertex_1
            elif new_vertex_2 not in visited_vertices:
                new_vertex = new_vertex_2
            else:
                print("There's an issue here")
                problem = True
        else:
            print("NOGOOD")
            # Make a mistake
            # Compute the probability distribution to choose the new vertex
            vertex_scores = cost_array[current_vertex, :]
            # Put a dummy value so that python does not complain
            vertex_scores[current_vertex] = 1
            distances = 1.0/vertex_scores
            for visited_idx in visited_vertices:
                distances[visited_idx] = 0
            distances = distances/distances.sum()

            new_vertex = np.random.choice(range(num_nodes), p=distances)

        # Add the new vertex to the partial tour and check if it was the optimal one
        visited_vertices.append(new_vertex)
        chose_optimal_arc = (forward_opt_tour[current_vertex]==new_vertex) or (backward_opt_tour[current_vertex]==new_vertex)

        if (not chose_optimal_arc) and (visited_vertices_nb+2<=num_nodes):
            print("ERREUR")

            partial_nodes_list = [node_idx for node_idx in nodes_list if node_idx not in visited_vertices[1:-1]]
            invert_partial_list = dict((val, key) for key,val in enumerate(partial_nodes_list))
            if problem==True:
                print(visited_vertices, len(visited_vertices))
                print("chose", chose_optimal_arc)
                print(partial_nodes_list)
                print(invert_partial_list)
            
            tmp_cost_array = np.delete(np.delete(cost_array, visited_vertices[1:-1], axis=0), visited_vertices[1:-1], axis=1)
            partial_cost_array = np.zeros((tmp_cost_array.shape[0]+1, tmp_cost_array.shape[1]+1))
            partial_cost_array[:-1, :-1] = tmp_cost_array[:, :]
            upper_bound = num_nodes*tmp_cost_array.max()
            partial_cost_array[-1, 1:] = upper_bound
            partial_cost_array[1:, -1] = upper_bound
            partial_cost_array[-1, -1] = 0            
            partial_cost_array[invert_partial_list[visited_vertices[-1]], -1] = 0
            partial_cost_array[-1, invert_partial_list[visited_vertices[-1]]] = 0
            # Prepare for the or tools optimization
            inflating_param = 1e3
            # Optimize
            tsp_file_path = "concorde_instances/temp_instance.tsp"
            create_concorde_input(tsp_file_path, partial_cost_array*inflating_param)
            solution_path = call_concorde_solver(tsp_file_path)
            partial_tour = retrieve_concorde_output(solution_path)
            # print(partial_tour)
            dummy_vertex_idx = partial_cost_array.shape[0]-1
            partial_tour = np.append(partial_tour[partial_tour!=dummy_vertex_idx], 0)
            partial_array = np.array(partial_nodes_list)
            # Update the optimal solution
            forward_opt_tour = {}
            backward_opt_tour = {}
            for tour_idx in range(len(partial_tour)-1):
                forward_opt_tour[partial_array[partial_tour][tour_idx]] = partial_array[partial_tour][tour_idx+1]
                backward_opt_tour[partial_array[partial_tour][tour_idx+1]] = partial_array[partial_tour][tour_idx]

        elif chose_optimal_arc and (current_vertex!=visited_vertices[0]):
            print("PAS ERREUR")
            pred_vertex = backward_opt_tour[current_vertex]
            next_vertex = forward_opt_tour[current_vertex] 
            del forward_opt_tour[current_vertex]
            del backward_opt_tour[current_vertex]
            forward_opt_tour[pred_vertex] = next_vertex
            backward_opt_tour[next_vertex] = pred_vertex

        else:
            print("OK QUOI")

        # Making the current vertex unaccessible for the next iterations
        current_vertex = new_vertex
        if args.display:
            display_current_situation(cost_array, coords_array, visited_vertices, num_nodes, forward_opt_tour)

    if args.display:
        plt.show()
    return np.array(visited_vertices)

########### Parameters ###########
# precision_values = np.arange(0.5, 1.001, 0.01)
precision_values = np.arange(0.3, 0.61, 0.01)
# precision_values = np.arange(0.94, 1.00001, 0.01)
##################################

for oracle_precision in precision_values:
    print("Exploring with precision : " + str(oracle_precision))
    test_dataset = DataReader(args.data, args.solution)

    total_length = 0
    for graph_idx in tqdm(range(test_dataset.num_graphs)):
        cost_array, coords_array, forward_opt_tour, backward_opt_tour = test_dataset.get_next_graph()
        # Change diagonal values to prevent staying at the same vertex
        # for node_idx in range(test_dataset.num_nodes):
        #     cost_array[node_idx, node_idx] = 10

        # Computing a tour based on the oracle
        tour_nodes = compute_tour_nodes(cost_array, test_dataset.num_nodes, \
                                        oracle_precision, coords_array, \
                                        forward_opt_tour, backward_opt_tour)
        tour_length = compute_mean_tour_length(test_dataset.num_graphs, tour_nodes, cost_array)
        total_length += tour_length

    print("Mean tour length : " + str(total_length/test_dataset.num_graphs))