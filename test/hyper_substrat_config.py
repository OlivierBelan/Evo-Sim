import numpy as np
import matplotlib.pyplot as plt
from typing import List
from typing import Tuple


def plot_points(input_coordinates, hidden_coordinates, output_coordinates):
    plt.figure(figsize=(6,6))

    # plot the points and specify the color for each set
    plt.scatter(input_coordinates[:,0], input_coordinates[:,1], color='r', label='Input')
    plt.scatter(hidden_coordinates[:,0], hidden_coordinates[:,1], color='g', label='Hidden')
    plt.scatter(output_coordinates[:,0], output_coordinates[:,1], color='b', label='Output')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.show()

def generate_vertical_line_points(inputs:int, outputs:int, hiddens:List[int]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    # Set the base for each line
    input_base:float = 0.15
    output_base:float = 0.85
    layers:int = len(hiddens)

    # Generate points for each line separately
    input_coordinates:np.ndarray  = np.column_stack((np.full(inputs, input_base), np.linspace(0, 1, inputs)))
    output_coordinates:np.ndarray = np.column_stack((np.full(outputs, output_base), np.linspace(0, 1, outputs)))

    # Calculate the hidden base increment
    hidden_base_increment = (output_base - input_base) / (layers + 1)

    # Generate coordinates for hidden lines
    hidden_coordinates = np.empty((0, 2))
    hidden_coordinates_list = []
    for i in range(layers):
        # Calculate the base of this hidden line
        hidden_base = input_base + (i+1) * hidden_base_increment

        # Generate the coordinates for this hidden layer
        hidden_layer_coordinates = np.column_stack((np.full(hiddens[i], hidden_base), np.linspace(0, 1, hiddens[i])))
        hidden_coordinates = np.append(hidden_coordinates, hidden_layer_coordinates, axis=0)
        hidden_coordinates_list.append(hidden_layer_coordinates)
    
    # return input_coordinates, output_coordinates, hidden_coordinates, hidden_coordinates_list
    return input_coordinates, output_coordinates, hidden_coordinates_list


def generate_multi_layer_circle_points(inputs:int, outputs:int, hiddens:List[int]) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    # Set the radius for each circle
    input_radius:float = 0.15
    output_radius:float = 0.45
    layers:int = len(hiddens)

    # Generate points for each circle separately
    t_input:np.ndarray  = np.linspace(0, 2*np.pi, inputs, endpoint=False)
    t_output:np.ndarray = np.linspace(0, 2*np.pi, outputs, endpoint=False)

    # Generate coordinates for each circle
    input_coordinates:np.ndarray  = np.column_stack((0.5 + input_radius*np.cos(t_input), 0.5 + input_radius*np.sin(t_input)))
    output_coordinates:np.ndarray = np.column_stack((0.5 + output_radius*np.cos(t_output), 0.5 + output_radius*np.sin(t_output)))

    # Calculate the hidden radius increment
    hidden_radius_increment = (output_radius - input_radius) / (layers + 1)

    # Generate coordinates for hidden circles
    hidden_coordinates = np.empty((0, 2))
    hidden_coordinates_list = []
    for i in range(layers):
        # Calculate the radius of this hidden layer
        hidden_radius = input_radius + (i+1) * hidden_radius_increment

        # Generate the coordinates for this hidden layer
        t_hidden = np.linspace(0, 2*np.pi, hiddens[i], endpoint=False)
        hidden_layer_coordinates = np.column_stack((0.5 + hidden_radius*np.cos(t_hidden), 0.5 + hidden_radius*np.sin(t_hidden)))
        hidden_coordinates = np.append(hidden_coordinates, hidden_layer_coordinates, axis=0)
        hidden_coordinates_list.append(hidden_layer_coordinates)

    # return input_coordinates, output_coordinates, hidden_coordinates, hidden_coordinates_list
    return input_coordinates, output_coordinates, hidden_coordinates_list

def main():
    # input_coordinates, output_coordinates, hidden_coordinates, hidden_coordinates_list = generate_vertical_line_points(13, 3, [20, 10, 10, 20])
    input_coordinates, output_coordinates, hidden_coordinates_list = generate_multi_layer_circle_points(13, 12, [20, 10, 10, 20])
    hidden_coordinates = np.array([pair for sublist in hidden_coordinates_list for pair in sublist], dtype=np.float32)
    
    print("input_coordinates: \n", input_coordinates)
    print("hidden_coordinates: \n", hidden_coordinates_list)
    print("hidden_coordinates: \n", hidden_coordinates)
    print("output_coordinates: \n", output_coordinates)
    plot_points(input_coordinates, hidden_coordinates, output_coordinates)
    # print("test_neat_RL")
    # neuro_evo_matrix_func()

if __name__ == "__main__":
    main()
