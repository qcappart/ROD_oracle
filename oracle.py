import os
import json
import argparse
import time

import math
import numpy as np
from tqdm import tqdm

from concorde.tsp import TSPSolver
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from config import *
from utils.google_tsp_reader import InferenceGoogleTSPReader
from utils.load_LKH_tours import load_LKH_predictions

parser = argparse.ArgumentParser(description='gcn_tsp_parser')
parser.add_argument('-c','--config', type=str, default="configs/default.json")
parser.add_argument('--tours_file', type=str, required=True)
parser.add_argument('--recalibrate', action='store_true')
parser.add_argument('--display', action='store_true')

args = parser.parse_args()
config_path = args.config

# Load config
config = get_config(config_path)
print("Loaded {}:\n{}".format(config_path, config))

dtypeFloat = torch.FloatTensor
dtypeLong = torch.LongTensor
torch.manual_seed(1)

def compute_mean_tour_length(bs_nodes, x_edges_values):
    total_length = 0
    graph_nb, nodes_nb, _ = x_edges_values.shape
    for graph_idx in range(graph_nb):
        # print(bs_nodes)
        for node_idx in range(nodes_nb-1):
            total_length += x_edges_values[graph_idx, \
                                           bs_nodes[graph_idx, node_idx], \
                                           bs_nodes[graph_idx, node_idx+1]]
        total_length += x_edges_values[graph_idx, \
                                       bs_nodes[graph_idx, 0], \
                                       bs_nodes[graph_idx, nodes_nb-1]]
    return total_length/graph_nb

def create_data(distance_array, inflating_param):
    data = {}
    data['distance_matrix'] = distance_array * inflating_param
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def ortools_solve(data):
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    # Create Routing managerModel.
    routing = pywrapcp.RoutingModel(manager)
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    # Extracting the tour
    index = routing.Start(0)
    partial_tour = [manager.IndexToNode(index)]
    while not routing.IsEnd(index):
        index = assignment.Value(routing.NextVar(index))
        partial_tour.append(manager.IndexToNode(index))
    # Removing the dummy vertex from the tour
    dummy_vertex_idx = data['distance_matrix'].shape[0]-1
    partial_tour.remove(dummy_vertex_idx)
    
    # lol_val = 0
    # for idx in range(len(partial_tour)-1):
    #     lol_val += data['distance_matrix'][partial_tour[idx], partial_tour[idx+1]]
    # print("ET WALAAAAAAAa : ", lol_val/1e5)

    return np.array(partial_tour)


def display_current_situation(x_edges_array, node_coords, visited_vertices, opt_tour, num_nodes):
    plt.plot(node_coords[0][:, 0], node_coords[0][:, 1], "bo")
    plt.plot(node_coords[0][visited_vertices, 0], node_coords[0][visited_vertices, 1], "b-")
    opt_current_vertex = visited_vertices[0]
    if (int(opt_tour[0, opt_current_vertex, :].argmax())==visited_vertices[-1]):
        opt_current_vertex = visited_vertices[-1]
    opt_tour_vertices = [opt_current_vertex]
    begin_vertex = opt_current_vertex
    begin_opt = True
    opt_tour_idx = 0
    while (begin_vertex != opt_current_vertex or begin_opt):
        begin_opt = False
        next_opt_vertex = int(opt_tour[0, opt_current_vertex, :].argmax())
        opt_tour_vertices.append(next_opt_vertex)
        opt_current_vertex = next_opt_vertex
        opt_tour_idx += 1
    opt_tour_vertices = opt_tour_vertices[:-1]
    # print(opt_tour)
    # print("chaines", visited_vertices, opt_tour_vertices)

    current_length = 0
    for idx in range(len(opt_tour_vertices)-1):
        current_length += x_edges_array[0, opt_tour_vertices[idx], opt_tour_vertices[idx+1]]
    for idx in range(len(visited_vertices)-1):
        current_length += x_edges_array[0, visited_vertices[idx], visited_vertices[idx+1]]

    plt.plot(node_coords[0][opt_tour_vertices, 0], node_coords[0][opt_tour_vertices, 1], "r-")
    plt.title(current_length)
    plt.pause(0.5)
    plt.clf()


def compute_bs_nodes(opt_tour, x_edges_array, recalibrate, num_nodes, batch_num, oracle_precision, best_arcs_percentage):
    # opt_tour_array = opt_tour.cpu().numpy()
    graph_bs_nodes = np.zeros(num_nodes)
    chose_optimal_arc = True
    current_vertex = 0
    visited_vertices_nb = 1
    visited_vertices = [0]
    nodes_list = range(num_nodes)
    for visited_vertices_nb in range(1, num_nodes):
    #while (visited_vertices_nb<num_nodes):
        # print(opt_tour.cpu().numpy()[0, current_vertex, :], opt_tour[0, :, current_vertex])
        
        problem = False

        if (np.random.rand()<=oracle_precision):
            # Looking at the vertices before and after and choosing the one not visited yet
            new_vertex_1 = np.argmax(opt_tour.cpu().numpy()[0, current_vertex, :])
            new_vertex_2 = np.argmax(opt_tour.cpu().numpy()[0, :, current_vertex])
            # print(new_vertex_1, new_vertex_2)
            if new_vertex_1 not in visited_vertices:
                new_vertex = new_vertex_1
            elif new_vertex_2 not in visited_vertices:
                new_vertex = new_vertex_2
            else:
                print("There's an issue here")
                print(visited_vertices_nb, current_vertex, new_vertex_1, new_vertex_2)
                print(opt_tour)
                print(opt_tour.cpu().numpy()[0, current_vertex, :])
                print(opt_tour.cpu().numpy()[0, :, current_vertex])
                problem = True
            # print("oui")
        else:
            ################## INVERSE PROBABILITY ##################
            distances = 1.0/x_edges_array[0, current_vertex, :]
            for visited_idx in visited_vertices:
                distances[visited_idx] = 0
            distances = distances/distances.sum()
            # print(distances)
            #########################################################

            ################## SOFTMAX PROBABILITY ##################
            # distances = np.exp(-x_edges_array[0, current_vertex, :])
            # for visited_idx in visited_vertices:
            #     distances[visited_idx] = 0
            # distances = distances/distances.sum()
            # # print(distances)
            #########################################################

            ################## LOG PROBABILITY ##################
            # distances = np.log(x_edges_array[0, current_vertex, :]/np.sqrt(2))
            # for visited_idx in visited_vertices:
            #     distances[visited_idx] = 0
            # distances = distances/distances.sum()
            # print(distances)
            #####################################################

            new_vertex = np.random.choice(range(num_nodes), p=distances)
            # print("non", distances)
        # print(new_vertex)

        visited_vertices.append(new_vertex)
        # print(visited_vertices)
        # print(current_vertex, max_proba_indices, new_vertex)
        chose_optimal_arc = int(opt_tour[0, current_vertex, new_vertex])==1 or int(opt_tour[0, new_vertex, current_vertex])==1

        # print(visited_vertices_nb, current_vertex, new_vertex, chose_optimal_arc, visited_vertices)

        if recalibrate and (not chose_optimal_arc) and (visited_vertices_nb+2<=num_nodes):
            partial_nodes_list = [node_idx for node_idx in nodes_list if node_idx not in visited_vertices[1:-1]]
            invert_partial_list = dict((val, key) for key,val in enumerate(partial_nodes_list))
            if problem==True:
                print(visited_vertices, len(visited_vertices))
                print("chose", chose_optimal_arc)
                print(partial_nodes_list)
                print(invert_partial_list)
            # print(partial_nodes_list, visited_vertices[-1])
            
            tmp_cost_array = np.delete(np.delete(x_edges_array, visited_vertices[1:-1], axis=1), visited_vertices[1:-1], axis=2)
            partial_cost_array = np.zeros((tmp_cost_array.shape[1]+1, tmp_cost_array.shape[2]+1))
            partial_cost_array[:-1, :-1] = tmp_cost_array[0, :, :]
            upper_bound = num_nodes*tmp_cost_array.max()
            partial_cost_array[-1, 1:] = upper_bound
            partial_cost_array[1:, -1] = upper_bound
            partial_cost_array[invert_partial_list[visited_vertices[-1]], -1] = 0
            partial_cost_array[-1, invert_partial_list[visited_vertices[-1]]] = 0
            # print(partial_cost_array)
            # Preparing for the or tools optimization
            inflating_param = 1e5
            data = create_data(partial_cost_array, inflating_param)
            # Optimize
            partial_tour = ortools_solve(data)
            partial_array = np.array(partial_nodes_list)
            #print("data ", partial_tour, partial_nodes_list, visited_vertices)
            #print("ortools final", partial_array[partial_tour])

            # Mettre a jour le opt_tour
            opt_tour = torch.zeros(opt_tour.shape)
            for tour_idx in range(len(partial_tour)-1):
                opt_tour[0, partial_array[partial_tour][tour_idx], partial_array[partial_tour][tour_idx+1]] = 1
            # the partial tour loops back to 0, so no need to add the closing link
        elif chose_optimal_arc:
            # print("CORRECTECTIO", visited_vertices, current_vertex, new_vertex, visited_vertices[-2])
            # print("opt_tour", opt_tour)
            opt_tour[0, visited_vertices[0], current_vertex] = 0
            opt_tour[0, current_vertex, visited_vertices[0]] = 0
            going_forward = True
            if opt_tour[0, current_vertex, new_vertex] == 0:
                going_forward = False
                opt_tour[0, new_vertex, current_vertex] = 0
            else:
                opt_tour[0, current_vertex, new_vertex] = 0
            if going_forward:
                opt_tour[0, visited_vertices[0], new_vertex] = 1
            else:
                opt_tour[0, new_vertex, visited_vertices[0]] = 1

        # Making the current vertex unaccessible for the next iterations
        graph_bs_nodes[visited_vertices_nb] = new_vertex
        current_vertex = new_vertex
        if args.display:
            display_current_situation(x_edges_array, batch.nodes_coord, visited_vertices, opt_tour, num_nodes)

    if args.display:
        plt.show()
    return graph_bs_nodes

num_nodes = config.num_nodes
num_neighbors = config.num_neighbors
batches_per_epoch = config.batches_per_epoch

batch_size = 1
filepath = config.test_filepath

########### Parameters ###########
# precision_values = np.arange(0.5, 1.001, 0.01)
precision_values = np.arange(0.86, 0.87, 0.01)
# precision_values = np.arange(0.94, 1.00001, 0.001)
best_arcs_percentage = 0.1
##################################

for oracle_precision in precision_values:
    print("Exploring with precision : " + str(oracle_precision))

    # Load data and apply blur
    dataset = InferenceGoogleTSPReader(num_nodes, num_neighbors, batch_size=batch_size, filepath=filepath)
    batches_per_epoch = dataset.max_iter
    dataset = iter(dataset)

    total_y_preds = load_LKH_predictions(args.tours_file)
    bs_nodes = np.zeros((batches_per_epoch, num_nodes), dtype=np.int32)
    total_length = 0

    for batch_num in tqdm(range(batches_per_epoch)):
        # Generate a batch of TSPs
        try:
            batch = next(dataset)
        except StopIteration:
            break

        x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), \
                        requires_grad=False)
        x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), \
                                requires_grad=False)
        # Changing diagonal values to prevent staying at the same vertex
        for node_idx in range(num_nodes):
            x_edges_values[0, node_idx, node_idx] = 10

        opt_tour = total_y_preds[batch_num:batch_num+1, :, :]
        x_edges_array = x_edges_values.cpu().numpy()

        # Computing a tour based on the oracle
        bs_nodes[batch_num, :] = compute_bs_nodes(opt_tour, x_edges_array, args.recalibrate, num_nodes, batch_num, oracle_precision, best_arcs_percentage)

        tour_length = compute_mean_tour_length(bs_nodes[batch_num, :].reshape((1, -1)), x_edges_values.cpu().numpy())
        
        total_length += tour_length

        # print("New instance")
        # if batch_num>10:
        #     break

    #print(bs_nodes)
    print("Mean tour length : " + str(total_length/batches_per_epoch))