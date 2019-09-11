import os
import subprocess
import numpy as np

def create_concorde_input(tsp_file_path, cost_array):
    # Concorde needs a integer array
    cost_array = cost_array.astype(int)
    num_nodes = cost_array.shape[0]
    with open(tsp_file_path, "w") as tsp_input_f:
        # Write the header
        tsp_input_f.write("NAME : temporary_tsp\n")
        tsp_input_f.write("TYPE : TSP\n")
        tsp_input_f.write("DIMENSION : " + str(num_nodes) + "\n")
        tsp_input_f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
        tsp_input_f.write("EDGE_WEIGHT_FORMAT : LOWER_DIAG_ROW\n")
        tsp_input_f.write("EDGE_WEIGHT_SECTION\n")
        # Write the edge weights
        for line_idx in range(0, num_nodes):
            line_str = np.array_str(cost_array[line_idx, :(line_idx+1)])
            line_str = line_str.replace("[", "").replace("]", "")
            tsp_input_f.write(line_str + "\n")
        tsp_input_f.write("EOF")
    return

def call_concorde_solver(instance_path):
    # Compose the command and ask to delete the temporary files
    command_line = ["./concorde/TSP/concorde", "-x"]
    # Put the output in a file
    solution_path = os.path.splitext(instance_path)[0] + ".sol"
    command_line += ["-o", solution_path]
    # Name the instance to solve
    command_line += [instance_path]
    # Call the executable but direct the output text to /dev/null
    FNULL = open(os.devnull, 'w')
    subprocess.call(command_line, stdout=FNULL, stderr=subprocess.STDOUT)
    return solution_path

def retrieve_concorde_output(solution_path):
    optimal_list = []
    with open(solution_path, "r") as tsp_output_f:
        result_lines = tsp_output_f.readlines()[1:]
        for result_line in result_lines:
            # Remove the line returns
            result_line = result_line.replace(" \n", "").replace("\n", "").split(" ")
            optimal_list += result_line
    # Turn the solution into a numpy array
    optimal_array = np.array(optimal_list, dtype=int)
    return optimal_array

if __name__=="__main__":
    np.random.seed(seed=3)
    test_size = 50
    test_array = np.random.randint(3, 10, (test_size, test_size))
    test_array[range(test_size), range(test_size)] = 0
    print(test_array)
    tsp_file_path = "concorde_instances/temp_instance.tsp"

    create_concorde_input(tsp_file_path, test_array)
    solution_path = call_concorde_solver(tsp_file_path)
    optimal_tour = retrieve_concorde_output(solution_path)
    print(optimal_tour)