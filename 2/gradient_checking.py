import matplotlib.pyplot as plt
import h5py
import numpy as np
from helpers import initialize_parameters_deep, L_model_forward, L_model_backward, update_parameters, compute_cost, predict

def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4"]:

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys


def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:245760].reshape((12288,20))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters


def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3", "dW3", "db4"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def check_gradients(parameters, grads, X, Y, epsilon=1e-7):
    parameters_values, _ = dictionary_to_vector(parameters)
    # TODO: vectorize grads
    grads_values = gradients_to_vector(grads)

    num_params = parameters_values.shape[0]
    J_plus = np.zeros((num_params, 1))
    J_minus = np.zeros((num_params, 1))
    gradapprox = np.zeros((num_params, 1))

    for i in range(num_params):
        # Compute costs for theta plus epsilon
        thetaplus = np.copy(parameters_values)
        thetaplus[0][i] = thetaplus[0][i] + epsilon
        new_AL, _ = L_model_forward(X, vector_to_dictionary(thetaplus))
        J_plus[i] = compute_cost(new_AL, Y)

        # Compute costs for theta minus epsilon
        thetaminus = np.copy(parameters_values)
        thetaminus[0][i] = thetaminus[0][i] - epsilon
        new_AL, _ = L_model_forward(X, vector_to_dictionary(thetaminus))
        J_minus[i] = compute_cost(new_AL, Y)

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (epsilon * 2.)


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    costs = []
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)
        check_gradients(parameters, grads, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.savefig("test.svg")

    return parameters


def train_cat_images(filename):
    with h5py.File(filename, 'r') as f:
        train_x_orig = f['parsed_images'][()]
        train_y = f['image_classes'][()]

    print("Number of training examples: " + str(train_x_orig.shape[0]))
    print("Number of training results: " + str(train_y.shape[0]))
    # flatten data
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    # normalize values between 0 and 1
    train_x = train_x_flatten / 255

    layers_dims = [12288, 20, 7, 5, 1]

    # the code puts train_y into a (1,train_x.shape[0]) np.array
    train_y = train_y.reshape(1, train_y.shape[0])

    parameters = L_layer_model(train_x, train_y, layers_dims, print_cost=True)
    predict(train_x, train_y, parameters)

    return parameters


parameters = train_cat_images("images.hdf5")
