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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from keras.layers import Dense, Dropout, SimpleRNN, GRU, LSTM

from utils import prepare_data
from rbflayer import RBFLayer, InitCentersRandom

# Mapping for Keras model types
model_mapping = {'RNN': SimpleRNN, 'LSTM': LSTM, 'GRU': GRU}


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
        poly = PolynomialFeatures(self.degree) #Tạo một đối tượng PolynomialFeatures từ thư viện Scikit-learn với mức độ của đa thức là self.degree.
        M = poly.fit_transform(X) #để biến đổi mảng đầu vào X thành một ma trận mới M sau khi đã được mở rộng đa thức
        return M #Trả về ma trận M đã được mở rộng đa thức.

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
                                st.write("Gradient is vanishing, break the loop")
                                break_loop = True
                        if not break_loop:
                            # Update model weights
                            weight = M_batch.T @ batch_y[:, i]
                            with np.errstate(all='raise'):
                                try:
                                    w[:, i] += np.linalg.inv(s) @ (weight - gradient)
                                except:
                                    st.write("Gradient is vanishing, break the loop")
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
    def create_model(model_name, t, unit, learning_rate) -> any:
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
        # Định nghĩ của mô hình
        model = Sequential([
            model_mapping[model_name](
                units=unit, return_sequences=True),
            Dropout(0.2),
            model_mapping[model_name](units=unit),
            Dropout(0.2), 
            Dense(units=t) #Định nghĩa một lớp dense (hoặc fully connected) với t đơn vị đầu ra.
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Tạo một bộ tối ưu hóa Adam bằng cách sử dụng tf.keras.optimizers.Adam() với một tốc độ học (learning rate) cụ thể.
        model.compile(optimizer=optimizer, loss='mse') #tối ưu hóa được định nghĩa trước optimizer và hàm mất mát được chọn là "mse" (mean squared error).

        return model

    def train(self) -> None:
        """
        Trains the deep learning model using the training dataset.
        """
        # Prepare training data
        # Chuẩn hóa dữ liệu training
        X_train, y_train = prepare_data(self.train_set, self.d, self.t)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Prepare validating data
        # Chuẩn hóa dữ liệu validating
        X_valid, y_valid = prepare_data(self.valid_set, self.d, self.t)
        X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

        # Create the model
        # Tạo model bằng create_model() của đối tượng hiện tại với các tham số:model_name=self.modelname:
        # Tên của mô hình được sử dụng.t=self.t: Số lượng đơn vị đầu ra của mô hình.unit=self.unit: Số lượng đơn vị (neuron) trong mỗi lớp mạng neural.learning_rate=0.001: 
        # Tốc độ học của thuật toán tối ưu hóa được sử dụng để cập nhật trọng số của mô hình.
        model = self.create_model(
            model_name=self.model_name, t=self.t, unit=self.unit, learning_rate=0.001)
        
        # Configure model checkpoint 
        # Tạo một đối tượng ModelCheckpoint để lưu trữ mô hình tốt nhất dựa trên giá trị của hàm mất mát trên tập validation (val_loss). 
        # monitor='val_loss': Đánh giá hiệu suất của mô hình dựa trên giá trị của hàm mất mát trên tập validation.mode='min':
        # Mô hình sẽ được lưu trữ khi giá trị của hàm mất mát trên tập validation là nhỏ nhất.verbose=1: 
        # Hiển thị thông báo khi mô hình được lưu trữ.save_best_only=True: Chỉ lưu trữ mô hình khi nó có hiệu suất tốt nhất trên tập validation.
        file_path = './best_model/best_model.keras'
        mc = ModelCheckpoint(file_path, monitor='val_loss',
                             mode='min', verbose=1, save_best_only=True), 

        # Compile and fit the model
        # Biên dịch và huân luyện mô hình
        # Huấn luyện mô hình bằng cách truyền dữ liệu huấn luyện và validation vào
        model.fit(X_train, y_train, epochs=self.epoch, batch_size=self.batch_size,
                  callbacks=[mc], shuffle=False, validation_data=(X_valid, y_valid))

        # Load best model
        # Load mô hình tốt nhất
        self.model = load_model(file_path)

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tests the trained model using the testing dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted values and actual values from the testing dataset.
        """
        # Prepare testing data
        # Chuẩn bị dữ liệu test
        X_test, y_test = prepare_data(self.test_set, self.d, self.t)

        # Predict using the model
        # Dự đoán bằng cách sử dụng mô hình
        predicted_values = np.array(self.model.predict(X_test))

        return predicted_values, y_test

    def hyperparameter(self) -> dict:
        # Prepare training data
        # Chuẩn bị dữ liệu huấn luyện
        X_train, y_train = prepare_data(self.train_set, self.d, self.t)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Prepare validating data
        # Chuẩn bị dữ liệu validation
        X_valid, y_valid = prepare_data(self.valid_set, self.d, self.t)
        X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

        # Define the search space
        #Định nghĩ hàm không gian tìm kiếm với 
        # unit: Giới hạn dưới là 16 và giới hạn trên là 128.
        #epoch: Giới hạn dưới là 50 và giới hạn trên là 100.
        #batch_size: Giới hạn dưới là 16 và giới hạn trên là 128.
        #learning_rate: Giới hạn dưới là 1e-4 và giới hạn trên là 1e-3.
        pbounds = {'unit': (16, 128),
                   'epoch': (50, 100),
                   'batch_size': (16, 128),
                   'learning_rate': (1e-4, 1e-3)}

        # Define the objective function
        # Định nghĩa hàm mục tiêu
        def objective(unit, epoch, batch_size, learning_rate):
            unit = int(unit) # Chuyển các đối số đầu vào (unit, epoch, batch_size, learning_rate) thành các kiểu dữ liệu số nguyên (int) hoặc số thực (float).
            epoch = int(epoch)
            batch_size = int(batch_size)
            learning_rate = float(learning_rate)

            # Build the RNN model
            # Xây dụng mô hình RNN
            model = self.create_model(self.model_name, self.t, unit, learning_rate)

            # Train the model
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(
                X_valid, y_valid), verbose=0)

            # Evaluate the model
            # Đánh giá mô hình bằng hàm loss
            loss = history.history['val_loss'][-1]

            return -loss  # Minimize the loss
                          # # Giảm thiểu hàm mất mát

        # Run Bayesian Optimization
        # Chạy tối ưu hóa Bayesian
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
        )

        # Maximize the objective function
        #Tối ưu hóa hàm mục tiêu
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
        if self.is_auto: #Kiểm tra xem có đang sử dụng chế độ tự động để xác định order của mô hình ARIMA hay không.Nếu có thì
            trace_output = StringIO() #lưu trữ kết quả từ quá trình xác định order tự động của mô hình.
            with redirect_stdout(trace_output): #để chuyển hướng output từ quá trình xác định order vào trace_output.
                model = pm.auto_arima(    #Sử dụng auto_arima để tự động xác định thông số của mô hình ARIMA.
                    self.train_set, trace=True, suppress_warnings=True,
                    start_p=0, max_p=20, start_d=0, max_d=10, start_q=0, max_q=10)

            trace_lines = trace_output.getvalue().split('\n') #Chia kết quả từ trace_output thành các dòng riêng biệt.
            trace_data = [line.split()
                          for line in trace_lines if line.strip()] #Tạo một danh sách các dòng dữ liệu đã được phân tách thành các từ riêng biệt.
            trace_data = trace_data[1:-2] #Loại bỏ các dòng không cần thiết từ kết quả.
            substrings_to_remove = [":", "sec", "intercept"] #Chuỗi con cần loại bỏ khỏi các dòng dữ liệu.
            trace_data = [' '.join(filter(
                lambda x: x not in substrings_to_remove, string)) for string in trace_data] #Loại bỏ các chuỗi không cần thiết khỏi mỗi dòng dữ liệu.

            trace_df = [] #Khởi tạo một danh sách rỗng để lưu trữ dữ liệu.

            for item in trace_data: #Duyệt qua mỗi dòng dữ liệu đã được xử lý.
                parts = item.split(" ") #Phân tách mỗi dòng dữ liệu thành các phần riêng biệt.

                order = parts[0]
                aic = float(parts[1][4:].replace(",", ""))
                time = parts[2].replace("Time=", "") #Trích xuất thông tin về order, AIC (Akaike Information Criterion), và thời gian xác định.

                trace_df.append((order, aic, time)) #Thêm thông tin đã trích xuất vào danh sách trace_df.
            columns = ['Order', 'AIC', 'Time']
            trace_df = pd.DataFrame(trace_df, columns=columns) #Tạo DataFrame từ danh sách trace_df.

            self.trace_df = trace_df #Lưu DataFrame vào thuộc tính trace_df của đối tượng.

            summary = model.get_params()["order"]
            best_p, best_d, best_q = (
                int(summary[0]),
                int(summary[1]),
                int(summary[2]),
            ) #Gán các giá trị của order tốt nhất cho các biến best_p, best_d, và best_q sau khi chuyển chúng thành kiểu số nguyên.

            self.p_arima = best_p
            self.d_arima = best_d
            self.q_arima = best_q #Gán các giá trị order tốt nhất cho các thuộc tính p_arima, d_arima, và q_arima của đối tượng.
        else: #Nếu không sử dụng chế độ tự động
            model = pm.ARIMA(order=(self.p_arima, self.d_arima, self.q_arima)) #Khởi tạo mô hình ARIMA với order được chỉ định trước
        self.model = model #Gán mô hình ARIMA cho thuộc tính model của đối tượng.

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs forecasting on the test set using the trained ARIMA model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the predicted values and the actual values from the test set.
        """
        #dự báo trên tập dữ liệu kiểm tra bằng cách sử dụng mô hình ARIMA đã được huấn luyện
        predicted_list = [] #Khởi tạo một danh sách rỗng để lưu trữ các giá trị dự đoán.
        training_data = self.train_set.copy() #Sao chép tập dữ liệu huấn luyện để sử dụng trong quá trình dự đoán.
        training_data = training_data.values.reshape(
            training_data.shape[0], -1) #Chuyển đổi dữ liệu huấn luyện thành mảng hai chiều để phù hợp với định dạng đầu vào của mô hình ARIMA.
        for t in range(len(self.test_set)): #Duyệt qua mỗi điểm dữ liệu trong tập dữ liệu kiểm tra.
            model_fit = self.model.fit(training_data) #Điều chỉnh mô hình ARIMA với dữ liệu huấn luyện
            predicted_value = model_fit.predict(n_periods=1) #Dự đoán giá trị tiếp theo sử dụng mô hình đã được điều chỉnh
            predicted_list.append(predicted_value) #Thêm giá trị dự đoán vào danh sách predicted_list.
            training_data = np.concatenate(
                (training_data, self.test_set[t].reshape(1, -1))) #Cập nhật dữ liệu huấn luyện bằng cách thêm giá trị dự đoán mới vào cuối của nó.
        return np.array(predicted_list), np.array(self.test_set) #Trả về một tuple chứa danh sách các giá trị dự đoán và danh sách các giá trị thực tế từ tập dữ liệu kiểm tra.

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
    train_dl_set: np.ndarray = None
    test_set: np.ndarray = None
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
    arima_predict_set: np.array = None
    arima_fitted_set: np.array = None

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
            - The method calculates the weight based on the formula: w_parallel = sum((y1 - y2) * (y - y2)) / sum((y1 - y2) ** 2)
            - The weight is then constrained to be between 0 and 1 using the min and max functions.

        """
        y = np.array(self.train_set).copy().reshape(-1, 1) #Sao chép và chuyển đổi tập dữ liệu huấn luyện thành mảng numpy hai chiều.
        
        # st.write(y.shape)
        # st.write(y1.shape)
        # st.write(y2.shape)
        if len(y) != len(y1):
            y = y[len(y) - len(y1):] # Kiểm tra kích thước của dữ liệu huấn luyện phù hợp với dự đoán từ các mô hình.
        y = y[self.d if self.model1_name != "ARIMA" else self.p_arima:]
        y1 = y1[self.d if self.model1_name != "ARIMA" else self.p_arima:]
        y2 = y2[self.d if self.model1_name != "ARIMA" else self.p_arima:]  #Loại bỏ các giá trị ban đầu từ dữ liệu huấn luyện để phù hợp với dữ liệu được dự đoán từ mô hình.
        # st.write(y.shape)
        # st.write(y1.shape)
        # st.write(y2.shape)
        
        numerator = np.sum((y2 - y1) * (y - y1)) # tử só bằng  tích của sai số giữa dự đoán của mô hình thứ hai và mô hình thứ nhất với sai số giữa dự đoán của mô hình thứ nhất và dữ liệu thực tế.
        denominator = np.sum((y2 - y1) ** 2) # mẫu số bằng  tổng bình phương của sai số giữa dự đoán của mô hình thứ hai và mô hình thứ nhất.
        w_parallel = numerator / denominator if denominator != 0 else 0  #Tính trong số bằng công thức numerator / denominator nếu không trọng số gán bằng 0
        # Ensure w_parallel is between 0 and 1
        #Trọng số chỉ nằm trong khoản 0 đến 1
        w_parallel = max(0, min(w_parallel, 1))
        return w_parallel

    def calculate_errors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the training and testing errors for the combining model.
        """
        #Khởi tạo các biến để lưu trữ các sai số của quá trình huấn luyện 
        training_error = []
        testing_error = []
        validating_error = []

        if self.model1_name == "ARIMA": # Nếu là mô hình ARIMA
            self.model1 = ARIMAModel(
                self.train_set, self.test_set, self.p_arima, self.d_arima, self.q_arima, self.is_auto) #Tạo một đối tượng ARIMAModel với các thông số được cung cấp và lưu vào biến self.model1.
            self.model1.train() # huấn luyện mô hình ARIMA với dữ liệu đã có

            if self.is_auto: #Cập nhật các tham số ARIMA (nếu là tự động):
                self.p_arima = self.model1.p_arima
                self.d_arima = self.model1.d_arima
                self.q_arima = self.model1.q_arima

            if self.combine_strategy == "Tuần tự Cộng": # Nếu là mô hình ghép lai tuần tự Cộng
                if self.arima_fitted_set is None: # Nếu không có dữ liệu dự đoán ARIMA trước, ta cần huấn luyện mô hình ARIMA trên dữ liệu huấn luyện và dùng phương thức predict_in_sample() để dự đoán các giá trị.
                    fitted_values = self.model1.model.fit(
                        self.train_set).predict_in_sample()
                else: # Ngược lại
                    fitted_values = self.arima_fitted_set.copy() # sao chép dữ liệu đó vào biến fitted_values.
                    
                training_error = self.train_set - fitted_values # tính sai số huấn luyện bằng lấy hiệu giữa dữ liệu huấn luyện và các giá trị dự đoán từ mô hình ARIMA.

                if self.arima_predict_set is None:
                    predict, actual = self.model1.test()
                else:
                    predict, actual = np.array(self.arima_predict_set.copy()), np.array(self.test_set.copy())
                testing_error = actual.reshape(-1, 1) - predict
            else:
                if self.arima_fitted_set is None:
                    fitted_values = self.model1.model.fit(
                        self.train_set).predict_in_sample()
                else:
                    fitted_values = self.arima_fitted_set.copy()
                    
                training_error = self.train_set / fitted_values

                if self.arima_predict_set is None:
                    predict, actual = self.model1.test()
                else:
                    predict, actual = np.array(self.arima_predict_set.copy()), np.array(self.test_set.copy())
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
            self.model1 = DeepLearningModel(
                self.model1_name, self.train_set, self.test_set, self.valid_set, self.d, self.t, self.unit, self.epoch, self.batch_size)

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
        st.write
        return training_error, testing_error, validating_error

    def calculate_fitted_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the fitted values of model 1 and 2.
        """
        X, _ = prepare_data(self.train_set, self.d, self.t)
        if self.train_dl_set is not None:
            X_dl, _ = prepare_data(self.train_dl_set, self.d, self.t)

        if self.model1_name == "ARIMA":
            self.model1 = ARIMAModel(
                self.train_set, self.test_set, self.p_arima, self.d_arima, self.q_arima, self.is_auto)
            self.model1.train()

            if self.is_auto:
                self.p_arima = self.model1.p_arima
                self.d_arima = self.model1.d_arima
                self.q_arima = self.model1.q_arima

            if self.arima_fitted_set is None:
                model1_fitted_values = self.model1.model.fit(
                    self.train_set).predict_in_sample()
                self.arima_fitted_set = model1_fitted_values
                st.write("No")
            else:
                model1_fitted_values = self.arima_fitted_set.copy()
                st.write("Yes")

        elif self.model1_name == "Polynomial Classifier":
            self.model1 = PolynomialClassifierModel(
                self.train_set, self.test_set, self.d, self.t, self.degree, self.epoch, self.batch_size)
            self.model1.train()

            model1_fitted_values = np.dot(
                self.model1.poly_expansion(X), np.transpose(self.model1.model))
        elif self.model1_name == "RBFNN":
            self.model1 = RBFNNModel(self.train_dl_set, self.test_set, self.valid_set, self.d, self.t, self.unit, self.epoch, self.batch_size)
            self.model1.train()
            model1_fitted_values = np.array(self.model1.model.predict(X_dl))
        else:
            self.model1 = DeepLearningModel(
                self.model1_name, self.train_dl_set, self.test_set, self.valid_set, self.d, self.t, self.unit, self.epoch, self.batch_size)
            self.model1.train()

            model1_fitted_values = np.array(self.model1.model.predict(X_dl))

        if self.model2_name == "Polynomial Classifier":
            self.model2 = PolynomialClassifierModel(
                self.train_set, self.test_set, self.d, self.t, self.degree, self.epoch, self.batch_size)
            self.model2.train()

            model2_fitted_values = np.dot(
                self.model2.poly_expansion(X), np.transpose(self.model2.model))
        elif self.model2_name == "RBFNN":
            self.model2 = RBFNNModel(self.train_dl_set, self.test_set, self.valid_set, self.d, self.t, self.unit, self.epoch, self.batch_size)
            self.model2.train()
            model2_fitted_values = np.array(self.model2.model.predict(X_dl))
        else:
            self.model2 = DeepLearningModel(
                self.model2_name, self.train_dl_set, self.test_set, self.valid_set, self.d, self.t, self.unit, self.epoch, self.batch_size)
            self.model2.train()

            model2_fitted_values = np.array(self.model2.model.predict(X_dl))

        model1_fitted_values = np.array(model1_fitted_values)
        model2_fitted_values = np.array(model2_fitted_values)

        if len(model1_fitted_values) > len(model2_fitted_values):
            model1_fitted_values = model1_fitted_values[:len(model2_fitted_values)]
            
        return model1_fitted_values, model2_fitted_values

    def train_with_error_data(self, training_error: np.ndarray, testing_error: np.ndarray, validating_error: np.ndarray) -> None:
        if self.model1_name == "ARIMA":
            if self.model2_name == "Polynomial Classifier":
                self.model2 = PolynomialClassifierModel(
                    training_error, testing_error, self.p_arima, 1, self.degree, self.epoch, self.batch_size)
            elif self.model2_name == "RBFNN":
                self.model2 = RBFNNModel(training_error, testing_error, validating_error, self.d, self.t, self.unit, self.epoch, self.batch_size)
            else:
                self.model2 = DeepLearningModel(
                    self.model2_name, training_error, testing_error, validating_error, self.p_arima, 1, self.unit, self.epoch, self.batch_size
                )
        else:
            if self.model2_name == "Polynomial Classifier":
                self.model2 = PolynomialClassifierModel(
                    training_error, testing_error, self.d, self.t, self.degree, self.epoch, self.batch_size)
            elif self.model2_name == "RBFNN":
                self.model2 = RBFNNModel(training_error, testing_error, validating_error, self.d, self.t, self.unit, self.epoch, self.batch_size)
            else:
                self.model2 = DeepLearningModel(
                    self.model2_name, training_error, testing_error, validating_error, self.d, self.t, self.unit, self.epoch, self.batch_size)

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
            if self.model1_name == "ARIMA" and self.arima_predict_set is not None:
                prediction_value = self.arima_predict_set[self.d if self.model1_name != "ARIMA" else self.p_arima:]
            else:
                prediction_value = self.model1.test(
                )[0][self.d if self.model1_name != "ARIMA" else self.p_arima:]
            prediction_error = self.model2.test()[0]
            st.write("++", len(prediction_value), len(prediction_error))
            predict = prediction_value + prediction_error
        elif self.combine_strategy == "Tuần tự Nhân":
            if self.model1_name == "ARIMA" and self.arima_predict_set is not None:
                prediction_value = self.arima_predict_set[self.d if self.model1_name != "ARIMA" else self.p_arima:]
            else:
                prediction_value = self.model1.test(
                )[0][self.d if self.model1_name != "ARIMA" else self.p_arima:]
            prediction_error = self.model2.test()[0]
            
            st.write("**", len(prediction_value), len(prediction_error))
            
            predict = prediction_value * prediction_error
        else:
            if self.model1_name == "ARIMA" and self.arima_predict_set is not None:
                y1 = self.arima_predict_set[self.d if self.model1_name != "ARIMA" else self.p_arima:]
                st.write("Lets go")
            else:
                y1 = self.model1.test()[
                    0][self.d if self.model1_name != "ARIMA" else self.p_arima:]
                st.write("Nah")
            y2 = self.model2.test()[0]
            st.write("\\", len(y1), len(y2))

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
        model = Sequential() # tạo mô hình tuần tự
        # tạo một lớp RBF (Radial Basis Function) 
        #unit: Số lượng đơn vị trong lớp RBF, tức là số lượng hàm cơ sở.
        #initializer: Một trình khởi tạo để khởi tạo trọng số của lớp RBF. Trong trường hợp này, InitCentersRandom(X_train) được sử dụng để khởi tạo các trung tâm ngẫu nhiên từ dữ liệu huấn luyện.
        #betas: Tham số beta trong hàm RBF
        #input_shape: Hình dạng đầu vào của mô hình, trong trường hợp này là kích thước của các mẫu dữ liệu đầu vào.
        rbflayer = RBFLayer(unit,
                            initializer=InitCentersRandom(X_train),
                            betas=2.0,
                            input_shape=(X_train.shape[1],))
        model.add(rbflayer) #thêm lớp RBF vào mô hình
        model.add(Dense(t)) #thêm lớp Dense với t đơn vị vào mô hình. Lớp đầu ra của mạng.
        model.compile(loss='mean_squared_error', 
                      optimizer=tf.keras.optimizers.RMSprop(learning_rate)) #: Sử dụng hàm mất mát là sai số bình phương trung bình,  Sử dụng thuật toán tối ưu hóa RMSprop với tốc độ học là learning_rate

        return model # mô hình được trả về

    def train(self) -> None:
        """
        Trains the deep learning model using the training dataset.
        """
        # Prepare training data
        #Chuẩn bị dữ liệu trainning
        X_train, y_train = prepare_data(self.train_set, self.d, self.t)

        # Prepare validating data
        #Chuẩn bị dữ liệu validating
        X_valid, y_valid = prepare_data(self.valid_set, self.d, self.t)

        # Create the model
        #Tạo mô đồ Đối số X_train là dữ liệu huấn luyện, 
        # t là số lượng đầu ra mong muốn, 
        # unit là số lượng đơn vị trong mỗi lớp của mô hình, 
        # và learning_rate là tỷ lệ học của bộ tối ưu hóa.
        model = self.create_model(
            X_train, t=self.t, unit=self.unit, learning_rate=0.001)
        # Configure model checkpoint
        file_path = './best_model/best_model.keras' #là đường dẫn đến nơi lưu trữ mô hình tốt nhất.
        #monitor='val_loss': Theo dõi sự thay đổi của hàm mất mát trên tập validation (val_loss).
        #mode='min': Mục tiêu là tối thiểu hóa giá trị của độ đo theo dõi (trong trường hợp này là val_loss).
        #verbose=1: Hiển thị thông báo khi lưu mô hình.
        #save_best_only=True: Chỉ lưu lại mô hình tốt nhất dựa trên giá trị của độ đo theo dõi.
        mc = ModelCheckpoint(file_path, monitor='val_loss',
                             mode='min', verbose=1, save_best_only=True),

        # Compile and fit the model
        # Biên dịch và huấn luyện mô hình
        #X_train và y_train là dữ liệu huấn luyện.
        # epochs=self.epoch: Số lượng lượt huấn luyện qua toàn bộ dữ liệu.
        # batch_size=self.batch_size: Số lượng mẫu được sử dụng trong mỗi lượt huấn luyện.
        #callbacks=[mc]: Sử dụng callback để lưu mô hình có hiệu suất tốt nhất trên tập validation.
        # shuffle=False: Không xáo trộn dữ liệu trong mỗi lượt huấn luyện.
        #validation_data=(X_valid, y_valid): Sử dụng tập validation để đánh giá hiệu suất của mô hình trong quá trình huấn luyện.
        model.fit(X_train, y_train, epochs=self.epoch, batch_size=self.batch_size,
                  callbacks=[mc], shuffle=False, validation_data=(X_valid, y_valid))

        # Load best model
        #Load mô hình tốt nhất
        self.model = load_model(file_path, custom_objects={
                                'RBFLayer': RBFLayer})

    def test(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tests the trained model using the testing dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted values and actual values from the testing dataset.
        """
        # Prepare testing data
        #Chuẩn hóa dữ liệu tets
        X_test, y_test = prepare_data(self.test_set, self.d, self.t)

        # Predict using the model
        #Chọn mô hình dự đoán
        predicted_values = np.array(self.model.predict(X_test))

        return predicted_values, y_test

    def hyperparameter(self) -> dict:
        # Prepare training data
        #Chuẩn hóa dữ liệu trainning
        X_train, y_train = prepare_data(self.train_set, self.d, self.t)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Prepare validating data
        # Chuẩn hóa dữ liệu validating
        X_valid, y_valid = prepare_data(self.valid_set, self.d, self.t)
        X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)

        # Define the search space
        #Khai báo pbounds với các tham số
        #unit': Phạm vi của đơn vị (unit) trong mạng nơ-ron, từ 16 đến 128.
        #epoch': Phạm vi của số vòng lặp huấn luyện (epoch), từ 50 đến 100.
        #'batch_size': Phạm vi của kích thước batch, từ 16 đến 128.
        #'learning_rate': Phạm vi của tốc độ học (learning rate), từ 0.0001 đến 0.001.
        pbounds = {'unit': (16, 128),
                   'epoch': (50, 100),
                   'batch_size': (16, 128),
                   'learning_rate': (1e-4, 1e-3)}

        # Define the objective function
        # Định nghĩa làm với các tham số unit, epoch, batch_size, và learning_rate được chuyển đổi thành kiểu dữ liệu phù hợp (int hoặc float) để sử dụng trong quá trình huấn luyện mô hình.
        def objective(unit, epoch, batch_size, learning_rate):
            unit = int(unit)
            epoch = int(epoch)
            batch_size = int(batch_size)
            learning_rate = float(learning_rate)

            # Build the RNN model
            # Chạy mô hình RNN với các thông số X_train , self.t, unit, learning_rate
            model = self.create_model(X_train, self.t, unit, learning_rate)

            # Train the model
            #Huấn luyện mô hình với các tham số
            #X_train, y_train: Dữ liệu huấn luyện.
            #epochs=epoch: Số lượng epoch (vòng lặp qua toàn bộ dữ liệu huấn luyện) được chỉ định bởi tham số epoch.
            #batch_size=batch_size: Kích thước batch (số lượng mẫu dữ liệu được sử dụng trong mỗi lần cập nhật trọng số) được chỉ định bởi tham số batch_size.
            #verbose=0: Không hiển thị thông tin tiến trình trong quá trình huấn luyện.
            history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(
                X_valid, y_valid), verbose=0)

            # Evaluate the model
            #đánh giá mô hình sự: Đo lường mất mát trên tập validation cuối cùng từ lịch sử huấn luyện của mô hình.
            loss = history.history['val_loss'][-1]

            return -loss  # Minimize the loss # Giảm thiểu mất mát

        # Run Bayesian Optimization
        # Tối ưu hóa Bayesian với các tham số sau:
        optimizer = BayesianOptimization(
            f=objective, #hàm mục tiêu cần tối ưu hóa.
            pbounds=pbounds,#: Định nghĩa không gian tìm kiếm cho các siêu tham số của hàm mục tiêu.
            random_state=42, #Đặt seed cho quá trình tối ưu hóa Bayesian để kết quả có thể được tái tạo.
        )

        # Maximize the objective function
        # Tối ưu hóa hàm mục tiêu
        optimizer.maximize(init_points=5, n_iter=20) #Tối đa hóa hàm mục tiêu bằng cách chạy tối ưu hóa Bayesian với 5 điểm ban đầu và 20 lần lặp.
        result = optimizer.res #Lấy kết quả từ quá trình tối ưu hóa
        
        return result 
