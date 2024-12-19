import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error

def fit_mlp(training_data, target_column, num_neurons, learning_rate, epochs_num, activation_fun, seed_to_set):
    """
    Fits an MLP model using TensorFlow.

    Parameters:
    - training_data: pandas DataFrame containing the training data.
    - target_column: str, name of the target column in the DataFrame.
    - num_neurons: int, number of neurons in the hidden layer.
    - learning_rate: float, learning rate for the optimizer.
    - epochs_num: int, number of epochs to train the model.
    - activation_fun: str, activation function for the hidden layer.
    - seed_to_set: int, random seed for reproducibility.

    Returns:
    - A trained TensorFlow model.
    """
    # Set random seed for reproducibility
    tf.random.set_seed(seed_to_set)

    # Split features and target
    X = training_data.drop(columns=[target_column])
    y = training_data[target_column]

    # Define the model
    model = Sequential([
        Dense(num_neurons, activation=activation_fun, input_shape=(X.shape[1],)),
        Dense(1)  # I am solving a regression model here with a continuous output
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mae', 'mse'], 
                  )  # Track both MAE and MSE
    
    # Train the model
    model.fit(X, y, epochs=epochs_num, batch_size=16, verbose = 0)

    return model


def sieve_fit_brulee_mlp_20(train, test, formula_mod, num_neurons, learning_rate, activation_fun, epochs_init, rounds, seed_basis):
    MAEs = []
    candidate = []
    print('I am the new function')

    # The first round is not competitive. The others are.
    MAEs.append([])
    for x in range(1, epochs_init ** (epochs_init - 1)):
        print(x)
        np.random.seed(seed_basis * x)
    
        # Fit the model using fit_mlp
        model = fit_mlp(training_data=train, 
                        target_column='y',  # Assuming 'y' is the target column
                        num_neurons=num_neurons,
                        learning_rate=learning_rate,
                        epochs_num=epochs_init,
                        activation_fun=activation_fun,
                        seed_to_set=seed_basis * x)
        
        # Predict and compute MAE
        pred_test = model.predict(test.drop(columns='y'))
        pred_train = model.predict(train.drop(columns='y'))
        
        mae_test = mean_absolute_error(test['y'], pred_test)
        mae_train = mean_absolute_error(train['y'], pred_train)
        
        MAE = mae_test + mae_train
        res_list = [seed_basis * x, MAE]
        MAEs[0].append(res_list)
    
    # Find the best candidate
    candidate.append(extract_lowest(MAEs[0], epochs_init ** (epochs_init - 2)))
    
    # for i in range(1, rounds):
    #     if i == rounds:
    #         MAEs.append([])
    #         for x in [item[0] for item in candidate[i - 1]]:
    #             np.random.seed(x)
                
    #             # Fit the model using fit_mlp
    #             model = fit_mlp(training_data=train, 
    #                             target_column='y',
    #                             num_neurons=num_neurons,
    #                             learning_rate=learning_rate,
    #                             epochs_num=epochs_init ** (rounds - (rounds - (i + 1))),
    #                             activation_fun=activation_fun,
    #                             seed_to_set=x)
                
    #             # Predict and compute MAE
    #             pred_test = model.predict(test.drop(columns='y'))
    #             pred_train = model.predict(train.drop(columns='y'))
                
    #             mae_test = mean_absolute_error(test['y'], pred_test)
    #             mae_train = mean_absolute_error(train['y'], pred_train)
                
    #             MAE = mae_test + mae_train
    #             res_list = [x, MAE, model]
    #             MAEs[i].append(res_list)
    #     else:
    #         MAEs.append([])
    #         for x in [item[0] for item in candidate[i - 1]]:
    #             np.random.seed(x)
                
    #             # Fit the model using fit_mlp
    #             model = fit_mlp(training_data=train, 
    #                             target_column='y',
    #                             num_neurons=num_neurons,
    #                             learning_rate=learning_rate,
    #                             epochs_num=epochs_init ** (rounds - (rounds - (i + 1))),
    #                             activation_fun=activation_fun,
    #                             seed_to_set=x)
                
    #             # Predict and compute MAE
    #             pred_test = model.predict(test.drop(columns='y'))
    #             pred_train = model.predict(train.drop(columns='y'))
                
    #             mae_test = mean_absolute_error(test['y'], pred_test)
    #             mae_train = mean_absolute_error(train['y'], pred_train)
                
    #             MAE = mae_test + mae_train
    #             res_list = [x, MAE]
    #             MAEs[i].append(res_list)
        
    #     # Find the best candidate for this round
    #     candidate.append(extract_lowest(MAEs[i], epochs_init ** (epochs_init - (i + 1))))
    
    return candidate[rounds]


def extract_lowest(my_list, n):
    if len(my_list) == 0:
        raise ValueError("The input list is empty.")
    
    # Extract the second elements from each pair in the list
    second_elements = [x[1] for x in my_list]
    
    # Find the indices of the n lowest elements
    lowest_indices = np.argsort(second_elements)[:n]
    
    # Extract the n elements from the original list
    selected_elements = [my_list[i] for i in lowest_indices]
    
    return selected_elements
