import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def fit_mlp(training_data, 
            target_column, 
            num_neurons, 
            learning_rate, 
            epochs_num, 
            activation_fun, 
            seed_to_set):
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
        Dense(1, activation=None)  # I am solving a regression model here with a continuos output, this means that for the moment we try with the normal no activation oputput function
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='mse',
              metrics=['mae', 'mse'])  # Track both MAE and MSE
    
    
    # Train the model
    model.fit(X, y, epochs=epochs_num, verbose=0)

    return model
