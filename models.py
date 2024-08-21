import warnings
from io import StringIO
from typing import Tuple
from dataclasses import dataclass
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pmdarima as pm
import streamlit as st
import tensorflow as tf
from bayes_opt import BayesianOptimization
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, SimpleRNN, GRU, LSTM
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

from utils import prepare_data
from rbflayer import RBFLayer, InitCentersRandom

# Mapping for Keras model types
model_mapping = {'RNN': SimpleRNN, 'LSTM': LSTM,
                 'GRU': GRU}


@dataclass
class PolynomialClassifierModel:
    """
    A Polynomial Classifier Model.

    Attributes:
        train_set (np.ndarray): The training dataset.
        test_set (np.ndarray): The testing dataset.
        d (int): The number of days used for prediction.
        t (int): The number of days to predict.
        degree (int): The degree of polynomial expansion.
        epoch (int): The number of training epochs.
        batch_size (int): The size of each training batch.
        model (np.ndarray): The trained model weights.

    Methods:
        poly_expansion(X: np.ndarray) -> np.ndarray:
            Perform polynomial expansion on the input array.

        train() -> None:
            Train the polynomial classifier model using batch gradient descent.

        test() -> Tuple[np.ndarray, np.ndarray]:
            Test the polynomial classifier model on the testing dataset.

    """
    train_set: np.ndarray
    test_set: np.ndarray
    d: int
    t: int
    degree: int
    epoch: int
    batch_size: int
    model: np.ndarray = None

    def poly_expansion(self, X: np.ndarray) -> np.ndarray:
        """
        Perform polynomial expansion on the input array.

        Parameters:
            X (np.ndarray): The input array to be expanded.

        Returns:
            np.ndarray: The expanded array after polynomial expansion.

        """
        poly = PolynomialFeatures(self.degree)
        M = poly.fit_transform(X)
        return M

    def train(self) -> None:
        """
        Train the polynomial classifier model using batch gradient descent.

        Parameters:
            None

        Returns:
            None
        """
        # Prepare the training data
        X, y = prepare_data(self.train_set, self.d, self.t)

        # Perform polynomial expansion
        M = self.poly_expansion(X)

        # Initialize model weights
        w = np.random.randn(M.shape[1], self.t) * \
            np.sqrt(2. / (M.shape[1] + self.t))

        break_loop = False
        for _ in range(self.epoch):
            if break_loop:
                break
            # Shuffle the data
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Check if the number of sections is <= 0
            if X_shuffled.shape[0] // self.batch_size <= 0:
                batches_X = [X_shuffled]  # Set batches_X to X_shuffled itself
                # Assuming y corresponds to the labels/targets
                batches_y = [y_shuffled]
            else:
                # Split shuffled data into batches
                batches_X = np.array_split(
                    X_shuffled, X_shuffled.shape[0] // self.batch_size)
                batches_y = np.array_split(
                    y_shuffled, X_shuffled.shape[0] // self.batch_size)

                for batch_X, batch_y in zip(batches_X, batches_y):
                    if break_loop:
                        break
                    # Perform polynomial expansion on batch data
                    M_batch = self.poly_expansion(batch_X)

                    # Calculate covariance matrix s
                    s = M_batch.T @ M_batch

                    # Add small value to diagonal of s for invertibility
                    if np.linalg.det(s) == 0:
                        s += 1e-3 * np.identity(M_batch.shape[1])

                    for i in range(self.t):
                        if break_loop:
                            break
                        with np.errstate(all='raise'):
                            try:
                                # Calculate gradient
                                gradient = M_batch.T @ (M_batch @ w[:, i])
                            except:
                                # If overflow encountered, break the epoch loop
                                print("Gradient is vanishing, break the loop")
                                break_loop = True
                        if not break_loop:
                            # Update model weights
                            weight = M_batch.T @ batch_y[:, i]
                            with np.errstate(all='raise'):
                                try:
                                    w[:, i] += np.linalg.inv(s) @ (weight - gradient)
                                except:
                                    break_loop = True
                                    
        # Store updated model weights
        self.model = [w.reshape(-1)]

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Test the polynomial classifier model on the testing dataset.

        Parameters:
            None

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted values and actual values from the testing dataset.
        """
        X, y = prepare_data(self.test_set, self.d, self.t)
        result = np.dot(self.poly_expansion(X), np.transpose(self.model))
        return result, y


@dataclass
class DeepLearningModel:
    """
    A class representing a deep learning model for time series prediction.

    Attributes:
        model_name (str): The type of cell used in the model.
        train_set (np.ndarray): The training dataset.
        test_set (np.ndarray): The testing dataset.
        valid_set (np.ndarray): The validation dataset.
        d (int): The number of days used for prediction.
        t (int): The number of days to predict.
        unit (int): The number of units in the model.
        epoch (int): The number of epochs for training.
        batch_size (int): The batch size for training.
        model (any): The trained deep learning model (RNN | LSTM | GRU).

    Methods:
        train(): Trains the deep learning model using the training dataset.
        test(): Tests the trained model using the testing dataset.
        create_model(): Creates a deep learning model based on the specified parameters.
        hyperparameter(): Performs hyperparameter optimization for the deep learning model.

    """
    model_name: str
    train_set: np.ndarray
    test_set: np.ndarray
    valid_set: np.ndarray
    d: int
    t: int
    unit: int
    epoch: int
    batch_size: int
    model: any = None

    @staticmethod
    def create_model(model_name: str, t: int, unit: int, learning_rate: float) -> any:
        """
        Creates a deep learning model based on the specified parameters.

        Parameters:
            model_name (str): The type of cell used in the model ('RNN', 'LSTM', 'GRU').
            t (int): The number of days to predict.
            unit (int): The number of units in the model.
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            any: The created deep learning model.

        """
        # Define model architecture
        model = Sequential([
            model_mapping[model_name](
                units=unit, return_sequences=True),
            Dropout(0.2),
            model_mapping[model_name](units=unit),
            Dropout(0.2),
            Dense(units=t)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def train(self) -> None:
        """
        Trains the deep learning model using the training dataset.
        """
        # Prepare training data
        X_train, y_train = prepare_data(self.train_set, self.d, self.t)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Prepare validating data
        X_valid, y_valid = prepare_data(self.valid_set, self.d, self.t)
        X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

        # Create the model
        model = self.create_model(
            model_name=self.model_name, t=self.t, unit=self.unit, learning_rate=0.001)
        # Configure model checkpoint
        file_path = './best_model/best_model.keras'
        mc = ModelCheckpoint(file_path, monitor='val_loss',
                             mode='min', verbose=1, save_best_only=True),

        # Compile and fit the model
        model.fit(X_train, y_train, epochs=self.epoch, batch_size=self.batch_size,
                  callbacks=[mc], shuffle=False, validation_data=(X_valid, y_valid))

        # Load best model
        self.model = load_model(file_path)

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tests the trained model using the testing dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted values and actual values from the testing dataset.
        """
        # Prepare testing data
        X_test, y_test = prepare_data(self.test_set, self.d, self.t)

        # Predict using the model
        predicted_values = np.array(self.model.predict(X_test))

        return predicted_values, y_test

    def hyperparameter(self) -> dict:
        # Prepare training data
        X_train, y_train = prepare_data(self.train_set, self.d, self.t)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Prepare validating data
        X_valid, y_valid = prepare_data(self.valid_set, self.d, self.t)
        X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

        # Define the search space
        pbounds = {'unit': (16, 128),
                   'epoch': (50, 100),
                   'batch_size': (16, 128),
                   'learning_rate': (1e-4, 1e-3)}

        # Define the objective function
        def objective(unit, epoch, batch_size, learning_rate):
            unit = int(unit)
            epoch = int(epoch)
            batch_size = int(batch_size)
            learning_rate = float(learning_rate)

            # Build the RNN model
            model = self.create_model(
                self.model_name, self.t, unit, learning_rate)

            # Train the model
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(
                X_valid, y_valid), verbose=0)

            # Evaluate the model
            loss = history.history['val_loss'][-1]

            return -loss  # Minimize the loss

        # Run Bayesian Optimization
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
        )

        # Maximize the objective function
        optimizer.maximize(init_points=5, n_iter=20)
        result = optimizer.res

        return result


@dataclass
class ARIMAModel:
    """
    ARIMAModel class represents a model for performing ARIMA (AutoRegressive Integrated Moving Average) time series forecasting.

    Attributes:
        train_set (np.ndarray): The training dataset.
        test_set (np.ndarray): The testing dataset.
        p_arima (int): The AR (AutoRegressive) order.
        d_arima (int): The I (Integrated) order.
        q_arima (int): The MA (Moving Average) order.
        is_auto (bool, optional): Flag indicating whether to automatically determine the best order. Defaults to False.
        model (pm.arima.arima.ARIMA, optional): The ARIMA model. Defaults to None.
        trace_df (pd.DataFrame, optional): A DataFrame containing the trace of auto ARIMA. Defaults to None.

    Methods:
        train(): Trains the ARIMA model using the specified order or automatically determines the best order.
        test(): Performs forecasting on the test set using the trained ARIMA model.

    """
    train_set: np.ndarray
    test_set: np.ndarray
    p_arima: int
    d_arima: int
    q_arima: int
    is_auto: bool = False
    model: pm.arima.arima.ARIMA = None
    trace_df: pd.DataFrame = None

    def train(self) -> None:
        """
        Trains the ARIMA model using the specified order or automatically determines the best order.
        """
        if self.is_auto:
            trace_output = StringIO()
            with redirect_stdout(trace_output):
                model = pm.auto_arima(
                    self.train_set, trace=True, suppress_warnings=True,
                    start_p=0, max_p=20, start_d=0, max_d=10, start_q=0, max_q=10)

            trace_lines = trace_output.getvalue().split('\n')
            trace_data = [line.split()
                          for line in trace_lines if line.strip()]
            trace_data = trace_data[1:-2]
            substrings_to_remove = [":", "sec", "intercept"]
            trace_data = [' '.join(filter(
                lambda x: x not in substrings_to_remove, string)) for string in trace_data]

            trace_df = []

            for item in trace_data:
                parts = item.split(" ")

                order = parts[0]
                aic = float(parts[1][4:].replace(",", ""))
                time = parts[2].replace("Time=", "")

                trace_df.append((order, aic, time))
            columns = ['Order', 'AIC', 'Time']
            trace_df = pd.DataFrame(trace_df, columns=columns)

            self.trace_df = trace_df

            summary = model.get_params()["order"]
            best_p, best_d, best_q = (
                int(summary[0]),
                int(summary[1]),
                int(summary[2]),
            )

            self.p_arima = best_p
            self.d_arima = best_d
            self.q_arima = best_q
        else:
            model = pm.ARIMA(order=(self.p_arima, self.d_arima, self.q_arima))
        self.model = model

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs forecasting on the test set using the trained ARIMA model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the predicted values and the actual values from the test set.
        """
        predicted_list = []
        training_data = self.train_set.copy()
        training_data = training_data.values.reshape(
            training_data.shape[0], -1)
        for t in range(len(self.test_set)):
            model_fit = self.model.fit(training_data)
            predicted_value = model_fit.predict(n_periods=1)
            predicted_list.append(predicted_value)
            training_data = np.concatenate(
                (training_data, self.test_set[t].reshape(1, -1)))
        return np.array(predicted_list), np.array(self.test_set)


@dataclass
class CombiningModel:
    """
    A class representing a combining model for time series prediction.

    Attributes:
        model1_name (str): The selected model for the first step of combination.
        model2_name (str): The selected model for the second step of combination.
        combine_strategy (str): The strategy used for combining the predictions.
        train_set (np.ndarray): The training dataset.
        test_set (np.ndarray): The testing dataset.
        valid_set (np.ndarray): The validation dataset.
        valid_size (int): The size of the validation dataset.
        d (int): The number of days used to predict.
        t (int): The number of days wanted to predict.
        degree (int): The degree of polynomial expansion.
        unit (int): The number of units in the deep learning model.
        epoch (int): The number of epochs for training.
        batch_size (int): The batch size for training.
        p_arima (int): The order of the ARIMA model.
        d_arima (int): The degree of differencing in the ARIMA model.
        q_arima (int): The order of the moving average component in the ARIMA model.
        is_auto (bool): Whether to automatically determine the order of the ARIMA model.
        model1 (any): The trained model for the first step of combination.
        model2 (any): The trained model for the second step of combination.
        w_parallel (np.ndarray): The weight for combining the predictions.

    Methods:
        calculate_parallel_weight(y1: np.ndarray, y2: np.ndarray) -> float:
            Calculates the weight for combining the predictions based on the parallel combination strategy.

        train() -> None:
            Trains the combining model using the selected models and combination strategy.

        test() -> Tuple[np.ndarray, np.ndarray]:
            Tests the trained combining model using the testing dataset.

        calculate_errors() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            Calculates the training, testing, and validation errors for the combining model.

        calculate_fitted_values() -> Tuple[np.ndarray, np.ndarray]:
            Calculates the fitted values of model 1 and model 2.

        train_with_error_data(training_error: np.ndarray, testing_error: np.ndarray, validating_error: np.ndarray) -> None:
            Trains the combining model using the training, testing, and validation errors.
    """

    model1_name: str
    model2_name: str
    combine_strategy: str
    train_set: np.ndarray
    test_set: np.ndarray
    valid_set: np.ndarray = None
    valid_size: int = None
    d: int = None  # Days used to predict
    t: int = None  # Days wanted to predict
    degree: int = None
    unit: int = None
    epoch: int = None
    batch_size: int = None
    p_arima: int = None
    d_arima: int = None
    q_arima: int = None
    is_auto: bool = False
    model1: any = None
    model2: any = None
    w_parallel: np.ndarray = None

    def create_model(self):
        pass

    def calculate_parallel_weight(self, y1: np.ndarray, y2: np.ndarray) -> float:
        """
        Calculates the weight for combining the predictions based on the parallel combination strategy.

        Parameters:
            y1 (np.ndarray): The predictions from the first model.
            y2 (np.ndarray): The predictions from the second model.

        Returns:
            float: The weight for combining the predictions.

        Notes:
            - The method assumes that the training dataset is stored in the 'train_set' attribute of the CombiningModel instance.
            - The method also assumes that the selected models and combination strategy are stored in the corresponding attributes of the CombiningModel instance.
            - The method calculates the weight based on the formula: w_parallel = sum((y2 - y1) * (y - y1)) / sum((y2 - y1) ** 2)
            - The weight is then constrained to be between 0 and 1 using the min and max functions.

        """
        y = np.array(self.train_set).copy().reshape(-1, 1)
        if len(y) != len(y1):
            y = y[len(y) - len(y1):]
        y = y[self.d if self.model1_name != "ARIMA" else self.p_arima:]
        y1 = y1[self.d if self.model1_name != "ARIMA" else self.p_arima:]
        y2 = y2[self.d if self.model1_name != "ARIMA" else self.p_arima:]

        numerator = np.sum((y2 - y1) * (y - y1))
        denominator = np.sum((y2 - y1) ** 2)
        w_parallel = numerator / denominator if denominator != 0 else 0
        # Ensure w_parallel is between 0 and 1
        w_parallel = max(0, min(w_parallel, 1))
        return w_parallel

    def calculate_errors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the training and testing errors for the combining model.
        """
        training_error = []
        testing_error = []
        validating_error = []

        if self.model1_name == "ARIMA":
            self.model1 = ARIMAModel(
                self.train_set, self.test_set, self.p_arima, self.d_arima, self.q_arima, self.is_auto)
            self.model1.train()

            if self.is_auto:
                self.p_arima = self.model1.p_arima
                self.d_arima = self.model1.d_arima
                self.q_arima = self.model1.q_arima

            if self.combine_strategy == "Tuần tự Cộng":
                fitted_values = self.model1.model.fit(
                    self.train_set).predict_in_sample()
                training_error = self.train_set - fitted_values

                predict, actual = self.model1.test()
                testing_error = actual.reshape(-1, 1) - predict
            else:
                fitted_values = self.model1.model.fit(
                    self.train_set).predict_in_sample()
                training_error = self.train_set / fitted_values

                predict, actual = self.model1.test()
                testing_error = actual.reshape(-1, 1) / predict

                training_error[np.isinf(training_error)] = 0
                testing_error[np.isinf(testing_error)] = 0

        elif self.model1_name == "Polynomial Classifier":
            self.model1 = PolynomialClassifierModel(
                self.train_set, self.test_set, self.d, self.t, self.degree, self.epoch, self.batch_size)

            self.model1.train()

            X, y = prepare_data(self.train_set, self.d, self.t)
            result = np.dot(self.model1.poly_expansion(
                X), np.transpose(self.model1.model))

            if self.combine_strategy == "Tuần tự Cộng":
                training_error = y - result

                predict, actual = self.model1.test()
                testing_error = actual - predict
            else:
                training_error = y / result

                predict, actual = self.model1.test()
                testing_error = actual / predict

                training_error[np.isinf(training_error)] = 0
                testing_error[np.isinf(testing_error)] = 0

        else:
            self.model1 = DeepLearningModel(self.model1_name,
                                            self.train_set, self.test_set, self.valid_set, self.d, self.t, self.unit, self.epoch, self.batch_size)

            self.model1.train()

            if self.combine_strategy == "Tuần tự Cộng":
                X, y = prepare_data(self.train_set, self.d, self.t)
                result = np.array(self.model1.model.predict(X))
                training_error = y - result

                predict, actual = self.model1.test()
                testing_error = actual - predict
            else:
                X, y = prepare_data(self.train_set, self.d, self.t)
                result = np.array(self.model1.model.predict(X))
                training_error = y / result

                predict, actual = self.model1.test()
                testing_error = actual / predict

                training_error[np.isinf(training_error)] = 0
                testing_error[np.isinf(testing_error)] = 0

        if self.model2_name != "Polynomial Classifier":
            validating_error, training_error = train_test_split(
                self.train_set,
                train_size=self.valid_size / 100,
                shuffle=False
            )

        return training_error, testing_error, validating_error

    def calculate_fitted_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the fitted values of model 1 and 2.
        """
        X, _ = prepare_data(self.train_set, self.d, self.t)

        if self.model1_name == "ARIMA":
            self.model1 = ARIMAModel(
                self.train_set, self.test_set, self.p_arima, self.d_arima, self.q_arima, self.is_auto)
            self.model1.train()

            if self.is_auto:
                self.p_arima = self.model1.p_arima
                self.d_arima = self.model1.d_arima
                self.q_arima = self.model1.q_arima

            model1_fitted_values = self.model1.model.fit(
                self.train_set).predict_in_sample()

        elif self.model1_name == "Polynomial Classifier":
            self.model1 = PolynomialClassifierModel(
                self.train_set, self.test_set, self.d, self.t, self.degree, self.epoch, self.batch_size)
            self.model1.train()

            model1_fitted_values = np.dot(
                self.model1.poly_expansion(X), np.transpose(self.model1.model))
        else:
            self.model1 = DeepLearningModel(self.model1_name,
                                            self.train_set, self.test_set, self.valid_set, self.d, self.t, self.unit, self.epoch, self.batch_size)
            self.model1.train()

            model1_fitted_values = np.array(self.model1.model.predict(X))

        if self.model2_name == "Polynomial Classifier":
            self.model2 = PolynomialClassifierModel(
                self.train_set, self.test_set, self.d, self.t, self.degree, self.epoch, self.batch_size)
            self.model2.train()

            model2_fitted_values = np.dot(
                self.model2.poly_expansion(X), np.transpose(self.model2.model))
        else:
            self.model2 = DeepLearningModel(self.model2_name,
                                            self.train_set, self.test_set, self.valid_set, self.d, self.t, self.unit, self.epoch, self.batch_size)
            self.model2.train()

            model2_fitted_values = np.array(self.model2.model.predict(X))

        model1_fitted_values = np.array(model1_fitted_values)
        model2_fitted_values = np.array(model2_fitted_values)

        return model1_fitted_values, model2_fitted_values

    def train_with_error_data(self, training_error: np.ndarray, testing_error: np.ndarray, validating_error: np.ndarray) -> None:
        if self.model1_name == "ARIMA":
            if self.model2_name == "Polynomial Classifier":
                self.model2 = PolynomialClassifierModel(
                    training_error, testing_error, self.p_arima, 1, self.degree, self.epoch, self.batch_size)

            else:
                self.model2 = DeepLearningModel(self.model2_name,
                                                training_error, testing_error, validating_error, self.p_arima, 1, self.unit, self.epoch, self.batch_size
                                                )
        else:
            if self.model2_name == "Polynomial Classifier":
                self.model2 = PolynomialClassifierModel(
                    training_error, testing_error, self.d, self.t, self.degree, self.epoch, self.batch_size)
            else:
                self.model2 = DeepLearningModel(self.model2_name,
                                                training_error, testing_error, validating_error, self.d, self.t, self.unit, self.epoch, self.batch_size)

        self.model2.train()

    def train(self) -> None:
        """
        Trains the combining model using the selected models and combination strategy.
        """
        if self.combine_strategy != "Song song":
            training_error, testing_error, validating_error = self.calculate_errors()
            self.train_with_error_data(
                training_error, testing_error, validating_error)
        else:
            model1_fitted_values, model2_fitted_values = self.calculate_fitted_values()

            self.w_parallel = self.calculate_parallel_weight(
                model1_fitted_values, model2_fitted_values)

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tests the trained combining model using the testing dataset.
        """
        if self.combine_strategy == "Tuần tự Cộng":
            prediction_value = self.model1.test(
            )[0][self.d if self.model1_name != "ARIMA" else self.p_arima:]
            prediction_error = self.model2.test()[0]
            predict = prediction_value + prediction_error
        elif self.combine_strategy == "Tuần tự Nhân":
            prediction_value = self.model1.test(
            )[0][self.d if self.model1_name != "ARIMA" else self.p_arima:]
            prediction_error = self.model2.test()[0]
            prediction_value = prediction_value
            predict = prediction_value * prediction_error
        else:
            y1 = self.model1.test()[
                0][self.p_arima if self.model1_name == "ARIMA" else 0:]
            y2 = self.model2.test()[0]
            predict = self.w_parallel*y2 + (1-self.w_parallel)*y1

        actual = np.array(self.test_set)[len(self.test_set) - len(predict):]

        return predict, actual


@dataclass
class RBFNNModel:
    """
    RBFNNModel class represents a Radial Basis Function Neural Network model.

    Attributes:
        train_set (np.ndarray): The training dataset.
        test_set (np.ndarray): The testing dataset.
        valid_set (np.ndarray): The validation dataset.
        d (int): The number of input features.
        t (int): The number of output targets.
        unit (int): The number of units in the RBF layer.
        epoch (int): The number of epochs for training.
        batch_size (int): The batch size for training.
        model (any): The trained model.

    Methods:
        create_model(X_train: np.ndarray, t: int, unit: int, learning_rate: float) -> any:
            Creates and compiles the RBFNN model.

        train() -> None:
            Trains the RBFNN model using the training dataset.

        test() -> Tuple[np.ndarray, np.ndarray]:
            Tests the trained model using the testing dataset.

        hyperparameter() -> dict:
            Performs hyperparameter optimization using Bayesian Optimization.
    """
    train_set: np.ndarray
    test_set: np.ndarray
    valid_set: np.ndarray
    d: int
    t: int
    unit: int
    epoch: int
    batch_size: int
    model: any = None

    @staticmethod
    def create_model(X_train: np.ndarray, t: int, unit: int, learning_rate: float) -> any:
        # Define model architecture
        model = Sequential()
        rbflayer = RBFLayer(unit,
                            initializer=InitCentersRandom(X_train),
                            betas=2.0,
                            input_shape=(X_train.shape[1],))
        model.add(rbflayer)
        model.add(Dense(t))

        model.compile(loss='mean_squared_error',
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate))

        return model

    def train(self) -> None:
        """
        Trains the deep learning model using the training dataset.
        """
        # Prepare training data
        X_train, y_train = prepare_data(self.train_set, self.d, self.t)

        # Prepare validating data
        X_valid, y_valid = prepare_data(self.valid_set, self.d, self.t)

        # Create the model
        model = self.create_model(
            X_train, t=self.t, unit=self.unit, learning_rate=0.001)
        # Configure model checkpoint
        file_path = './best_model/best_model.keras'
        mc = ModelCheckpoint(file_path, monitor='val_loss',
                             mode='min', verbose=1, save_best_only=True),

        # Compile and fit the model
        model.fit(X_train, y_train, epochs=self.epoch, batch_size=self.batch_size,
                  callbacks=[mc], shuffle=False, validation_data=(X_valid, y_valid))

        # Load best model
        self.model = load_model(file_path, custom_objects={
                                'RBFLayer': RBFLayer})

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tests the trained model using the testing dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted values and actual values from the testing dataset.
        """
        # Prepare testing data
        X_test, y_test = prepare_data(self.test_set, self.d, self.t)

        # Predict using the model
        predicted_values = np.array(self.model.predict(X_test))

        return predicted_values, y_test

    def hyperparameter(self) -> dict:
        # Prepare training data
        X_train, y_train = prepare_data(self.train_set, self.d, self.t)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Prepare validating data
        X_valid, y_valid = prepare_data(self.valid_set, self.d, self.t)
        X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

        # Define the search space
        pbounds = {'unit': (16, 128),
                   'epoch': (50, 100),
                   'batch_size': (16, 128),
                   'learning_rate': (1e-4, 1e-3)}

        # Define the objective function
        def objective(unit, epoch, batch_size, learning_rate):
            unit = int(unit)
            epoch = int(epoch)
            batch_size = int(batch_size)
            learning_rate = float(learning_rate)

            # Build the RNN model
            model = self.create_model(X_train, self.t, unit, learning_rate)

            # Train the model
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(
                X_valid, y_valid), verbose=0)

            # Evaluate the model
            loss = history.history['val_loss'][-1]

            return -loss  # Minimize the loss

        # Run Bayesian Optimization
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
        )

        # Maximize the objective function
        optimizer.maximize(init_points=5, n_iter=20)
        result = optimizer.res

        return result
