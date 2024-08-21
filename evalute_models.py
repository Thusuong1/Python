import os
import json
from time import time
from io import StringIO
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pmdarima import auto_arima
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from data2 import Dataset
from rbflayer import RBFLayer
from utils import rescale_data, save_result, score, prepare_data
from models2 import PolynomialClassifierModel, DeepLearningModel, ARIMAModel, CombiningModel, RBFNNModel
######################################### Cấu hình streamlit #########################################
st.set_page_config(
    page_title="Dự báo",
    page_icon=":sparkles:",
)


######################################### Tham số khởi tạo #########################################
data_path = "../data/10_Tap_Du_Lieu/"
model_path = "./best_model/"
# file_names = ["arima.csv"]
# file_names = ["3_Gold_Price.csv", "8_Dau_Tho.csv", "9_Monthly_Beer.csv"]
file_names = ["1_weather.csv", "3_Gold_Price.csv", "8_Dau_Tho.csv", "9_Monthly_Beer.csv"]

standardize_strategy = "Min-max"
min_scale = 0
max_scale = 1

validate_size = 20
test_size = 20

DL = ["RNN", "LSTM", "GRU", "RBFNN"]
ARIMA_PC = ["ARIMA", "Polynomial Classifier"]
ARIMA_DL = ["ARIMA", "RNN", "LSTM", "GRU", "RBFNN"]
ARIMA_PC_DL = ["ARIMA", "Polynomial Classifier", "RNN", "LSTM", "GRU", "RBFNN"]
combines = ["Tuần tự Cộng", "Tuần tự Nhân", "Song song"]

d = 3
t = 1
degrees = 5
unit = 50
epoch = 100
batch_size = 64

######################################### Train và test tất cả mô hình #########################################
if st.button("ARIMA + PC", type="primary"):
    num_models = (len(ARIMA_PC) - 1 + degrees +
                  len(combines)*degrees)*len(file_names)
    with st.spinner("**Đang train và test tất cả ARIMA + PC**"):
        model_number = 0
        progress_bar = st.progress(0)
        st.write(f"**Tổng số mô hình: {num_models}**")

        for file_name in file_names:
            metric_table = None
            arima_predict_set = None
            arima_fitted_set = None
            arima_test_time = None
            arima_train_parallel_time = None

            data = Dataset()
            data.data = pd.read_csv(data_path + file_name)
            data.preprocess_data()
            data.target_data = data.data[data.data.columns[0]]
            data.standardize_data(standardize_strategy, min_scale, max_scale)
            data.split_data(test_size)

            # Auto-ARIMA model parameters
            model = auto_arima(data.train_data, trace=True, suppress_warnings=True,
                               start_p=0, max_p=20, start_d=0, max_d=10, start_q=0, max_q=10)
            summary = model.get_params()["order"]
            best_p, best_d, best_q = (
                int(summary[0]),
                int(summary[1]),
                int(summary[2]),
            )

            # Train và Test đơn lẻ một mô hình
            for model_name in ARIMA_PC:
                if metric_table is not None:
                    temp_metric_table = metric_table
                else:
                    temp_metric_table = None

                if model_name == "ARIMA":
                    model_number += 1
                    progress_percent = int((model_number / num_models) * 100)
                    progress_bar.progress(progress_percent)

                    model = ARIMAModel(
                        data.train_data, data.test_data, best_p, best_d, best_q)

                    # Train
                    start_time = time()
                    model.train()
                    train_time = float("{:.2f}".format(time() - start_time))

                    # Test
                    start_time = time()
                    predict, actual = model.test()
                    test_time = float("{:.2f}".format(time() - start_time))

                    arima_predict_set = predict
                    arima_test_time = test_time

                    # Score
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                    metrics.update({"Thời gian train": train_time,
                                    "Thời gian test": test_time})
                    temp_metric_table = pd.DataFrame(
                        metrics, index=[model_name])
                    if temp_metric_table is not None:
                        merge = [metric_table, temp_metric_table]
                        metric_table = pd.concat(merge)

                else:
                    for degree in range(1, degrees + 1):
                        model_number += 1
                        progress_percent = int(
                            (model_number / num_models) * 100)
                        progress_bar.progress(progress_percent)

                        model = PolynomialClassifierModel(
                            data.train_data, data.test_data, d, t, degree, epoch, batch_size)

                        # Train
                        start_time = time()
                        model.train()
                        train_time = float(
                            "{:.2f}".format(time() - start_time))

                        # Test
                        start_time = time()
                        predict, actual = model.test()
                        test_time = float("{:.2f}".format(time() - start_time))

                        # Score
                        predict_inv = rescale_data(
                            data.target_data, predict, standardize_strategy)
                        actual_inv = rescale_data(
                            data.target_data, actual, standardize_strategy)
                        metrics = score(predict_inv, actual_inv)
                        metrics.update({"Thời gian train": train_time,
                                        "Thời gian test": test_time})
                        temp_metric_table = pd.DataFrame(
                            metrics, index=[f"{model_name} bậc {degree}"])
                        if temp_metric_table is not None:
                            merge = [metric_table, temp_metric_table]
                            metric_table = pd.concat(merge)

            # Train và Test mô hình kết hợp
            for degree in range(1, degrees + 1):
                for combine in combines:
                    model_number += 1
                    progress_percent = int((model_number / num_models) * 100)
                    progress_bar.progress(progress_percent)

                    if metric_table is not None:
                        temp_metric_table = metric_table
                    else:
                        temp_metric_table = None

                    model = CombiningModel("ARIMA", "Polynomial Classifier", combine, data.train_data, None, data.test_data, None, None,
                                           best_p, 1, degree, unit, epoch, batch_size,
                                           best_p, best_d, best_q, False, None, None, None,
                                           arima_predict_set, arima_fitted_set)
                    # Train
                    start_time = time()
                    model.train()
                    train_time = float("{:.2f}".format(time() - start_time))

                    if arima_fitted_set is None:
                        arima_fitted_set = model.arima_fitted_set
                    if combine == "Song song":
                        if arima_train_parallel_time is None:
                            arima_train_parallel_time = train_time
                        else:
                            train_time += arima_train_parallel_time

                    # Test
                    start_time = time()
                    predict, actual = model.test()
                    test_time = float("{:.2f}".format(
                        time() - start_time)) + arima_test_time

                    # Score
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                    metrics.update({"Thời gian train": train_time,
                                    "Thời gian test": test_time})
                    temp_metric_table = pd.DataFrame(
                        metrics, index=[f"ARIMA {combine} Polynomial Classifier bậc {degree}"])
                    if temp_metric_table is not None:
                        merge = [metric_table, temp_metric_table]
                        metric_table = pd.concat(merge)

            params = ["dataset_name", "column_name", "d", "t",
                      "split", "train_size", "test_size",
                      "standardize", "degree", "unit", "epoch", "batch_size",
                      "p_arima", "d_arima", "q_arima"]
            values = [file_name, data.data.columns[0], d, t,
                      f"{100 - test_size}/{test_size}", data.train_data.size, data.test_data.size,
                      standardize_strategy, degree, unit, epoch, batch_size, best_p, best_d, best_q]
            meanings = ["Tên tập dữ liệu", "Tên cột dự đoán", "Số ngày dùng để dự đoán", "Số ngày để dự đoán",
                        "Tỉ lệ tập train/test", "Kích thước tập train", "Kích thước tập test",
                        "Phương thức chuẩn hóa", "Số bậc đa thức", "Số node cho hidden layer",
                        "Số lượng epoch cho mỗi lần train", "Số lượng batch_size cho mỗi lần epoch",
                        "Số bậc tự hồi qui (AR)", "Số bậc sai phân (I)", "Số bậc trung bình trượt (MA)"]
            info_table = pd.DataFrame({
                "Tham số": params,
                "Giá trị": values,
                "Ý nghĩa": meanings
            })
            save_result(None, metric_table, info_table,
                        f'{file_name} ARIMA + PC')
            st.table(metric_table)
        st.success("**Tất cả mô hình đã được kiểm nghiệm!**")

if st.button("ARIMA + DL", type="primary"):
    num_models = (len(ARIMA_DL) + len(combines) *
                  (len(ARIMA_DL) - 1))*len(file_names)
    with st.spinner("**Đang train và test tất cả ARIMA + DL**"):
        model_number = 0
        progress_bar = st.progress(0)
        st.write(f"**Tổng số mô hình: {num_models}**")

        for file_name in file_names:
            metric_table = None
            arima_predict_set = None
            arima_fitted_set = None
            arima_test_time = None
            arima_train_parallel_time = None

            data = Dataset()
            data.data = pd.read_csv(data_path + file_name)
            data.preprocess_data()
            data.target_data = data.data[data.data.columns[0]]
            data.standardize_data(standardize_strategy, min_scale, max_scale)
            data.split_data(test_size, validate_size)

            # Auto-ARIMA model parameters
            trace_output = StringIO()
            with redirect_stdout(trace_output):
                model = auto_arima(
                    data.train_data, trace=True, suppress_warnings=True,
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
                time_run = parts[2].replace("Time=", "")

                trace_df.append((order, aic, time_run))
            columns = ['Order', 'AIC', 'Time']
            trace_df = pd.DataFrame(trace_df, columns=columns)

            summary = model.get_params()["order"]
            best_p, best_d, best_q = (
                int(summary[0]),
                int(summary[1]),
                int(summary[2]),
            )

            # Train và Test đơn lẻ một mô hình
            for model_name in ARIMA_DL:
                model_number += 1
                progress_percent = int((model_number / num_models) * 100)
                progress_bar.progress(progress_percent)

                if metric_table is not None:
                    temp_metric_table = metric_table
                else:
                    temp_metric_table = None

                if model_name == "ARIMA":
                    model = ARIMAModel(
                        data.train_data, data.test_data, best_p, best_d, best_q)

                    # Train
                    start_time = time()
                    model.train()
                    train_time = float("{:.2f}".format(time() - start_time))

                    # Test
                    start_time = time()
                    predict, actual = model.test()
                    test_time = float("{:.2f}".format(time() - start_time))

                    arima_predict_set = predict
                    arima_test_time = test_time

                    # Score
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                    metrics.update({"Thời gian train": train_time,
                                    "Thời gian test": test_time})
                    temp_metric_table = pd.DataFrame(
                        metrics, index=[model_name])
                    if temp_metric_table is not None:
                        merge = [metric_table, temp_metric_table]
                        metric_table = pd.concat(merge)

                else:
                    best_model_path_json = model_path + file_name + "/" + model_name + ".json"

                    with open(best_model_path_json, 'r') as json_file:
                        hyperparameters = json.load(json_file)

                    # Now you can access hyperparameters like units, epochs, batch_size, learning_rate
                    best_unit = hyperparameters['units']
                    best_epoch = hyperparameters['epochs']
                    best_batch_size = hyperparameters['batch_size']
                    best_learning_rate = hyperparameters['learning_rate']


                    # Prepare training data
                    X_train, y_train = prepare_data(data.train_data, d, t)
                    X_train_rbfnn = X_train.copy()
                    X_train = X_train.reshape(
                        X_train.shape[0], X_train.shape[1], 1)

                    # Prepare validating data
                    X_valid, y_valid = prepare_data(data.valid_data, d, t)
                    X_valid = X_valid.reshape(
                        X_valid.shape[0], X_valid.shape[1], 1)

                    if model_name == "RBFNN":
                        model = RBFNNModel(data.train_dl_data, data.test_data, data.valid_data,
                                           d, t, best_unit, best_epoch, best_batch_size)
                    else:
                        model = DeepLearningModel(model_name, data.train_dl_data, data.test_data, data.valid_data,
                                                d, t, best_unit, best_epoch, best_batch_size)

                    if model_name == "RBFNN":
                        hyper_model = model.create_model(
                            X_train_rbfnn, t, best_unit, best_learning_rate)
                    else:
                        hyper_model = model.create_model(
                            model_name, t, best_unit, best_learning_rate)
                        
                    best_model_path = model_path + file_name + "/" + model_name + ".keras"
                    mc = ModelCheckpoint(best_model_path, monitor='val_loss',
                                         mode='min', verbose=1, save_best_only=True),

                    # Train
                    start_time = time()

                    # Compile and fit the model
                    hyper_model.fit(X_train, y_train, epochs=best_epoch, batch_size=best_batch_size,
                                    callbacks=[mc], shuffle=False, validation_data=(X_valid, y_valid))
                    
                    train_time = float("{:.2f}".format(time() - start_time))
                    
                    if model_name == "RBFNN":
                        hyper_model = load_model(best_model_path, custom_objects={'RBFLayer': RBFLayer})
                    else:
                        hyper_model = load_model(best_model_path)


                    model.model = hyper_model

                    # Test
                    start_time = time()
                    predict, actual = model.test()
                    test_time = float("{:.2f}".format(
                        time() - start_time))

                    # Score
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                    metrics.update({"Thời gian train": train_time,
                                    "Thời gian test": test_time})
                    temp_metric_table = pd.DataFrame(
                        metrics, index=[model_name])
                    if temp_metric_table is not None:
                        merge = [metric_table, temp_metric_table]
                        metric_table = pd.concat(merge)

            # Train và Test mô hình kết hợp
            for model_name in ARIMA_DL[1:]:
                for combine in combines:
                    model_number += 1
                    progress_percent = int((model_number / num_models) * 100)
                    progress_bar.progress(progress_percent)

                    if metric_table is not None:
                        temp_metric_table = metric_table
                    else:
                        temp_metric_table = None

                    model = CombiningModel("ARIMA", model_name, combine, data.train_data, data.train_dl_data, data.test_data,
                                           data.valid_data, validate_size, best_p, 1, None, unit, epoch, batch_size,
                                           best_p, best_d, best_q, False, None, None, None,
                                           arima_predict_set, arima_fitted_set)

                    # Train
                    start_time = time()
                    model.train()
                    train_time = float("{:.2f}".format(time() - start_time))

                    if arima_fitted_set is None:
                        arima_fitted_set = model.arima_fitted_set
                    if combine == "Song song":
                        if arima_train_parallel_time is None:
                            arima_train_parallel_time = train_time
                        else:
                            train_time += arima_train_parallel_time

                    # Test
                    start_time = time()
                    predict, actual = model.test()
                    test_time = float("{:.2f}".format(
                        time() - start_time)) + arima_test_time

                    # Score
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                    metrics.update({"Thời gian train": train_time,
                                    "Thời gian test": test_time})
                    temp_metric_table = pd.DataFrame(
                        metrics, index=[f"ARIMA {combine} {model_name}"])
                    if temp_metric_table is not None:
                        merge = [metric_table, temp_metric_table]
                        metric_table = pd.concat(merge)

            params = ["dataset_name", "column_name", "d", "t",
                      "split", "train_size", "test_size", "validate_size",
                      "standardize", "p_arima", "d_arima", "q_arima"]
            values = [file_name, data.data.columns[0], d, t,
                      f"{100 - test_size}/{test_size}", data.train_data.size, data.test_data.size, data.valid_data.size,
                      standardize_strategy, best_p, best_d, best_q]
            meanings = ["Tên tập dữ liệu", "Tên cột dự đoán", "Số ngày dùng để dự đoán", "Số ngày để dự đoán",
                        "Tỉ lệ tập train/test", "Kích thước tập train", "Kích thước tập test", "Kích thước tập validate",
                        "Phương thức chuẩn hóa", "Số bậc tự hồi qui (AR)", "Số bậc sai phân (I)", "Số bậc trung bình trượt (MA)"]

            info_table = pd.DataFrame({
                "Tham số": params,
                "Giá trị": values,
                "Ý nghĩa": meanings
            })

            best_model_path_json = model_path + file_name + "/" + "RNN" + ".json"
            with open(best_model_path_json, 'r') as json_file:
                rnn_hyperparameter = json.load(json_file)
            best_model_path_json = model_path + file_name + "/" + "LSTM" + ".json"
            with open(best_model_path_json, 'r') as json_file:
                lstm_hyperparameter = json.load(json_file)
            best_model_path_json = model_path + file_name + "/" + "GRU" + ".json"
            with open(best_model_path_json, 'r') as json_file:
                gru_hyperparameter = json.load(json_file)
            # Create DataFrames from the loaded hyperparameters with reset index
            df_rnn = pd.DataFrame.from_dict(
                rnn_hyperparameter, orient='index', columns=['RNN']).T
            df_lstm = pd.DataFrame.from_dict(
                lstm_hyperparameter, orient='index', columns=['LSTM']).T
            df_gru = pd.DataFrame.from_dict(
                gru_hyperparameter, orient='index', columns=['GRU']).T
            # Concatenate the DataFrames
            hyperparameter_table = pd.concat([df_rnn, df_lstm, df_gru], axis=0)
            # Rename index
            hyperparameter_table.insert(0, 'Mô hình', ['RNN', 'LSTM', 'GRU'])

            save_result(None, metric_table, info_table,
                        f'{file_name} ARIMA + DL', hyperparameter_table, trace_df)

        st.success("**Tất cả mô hình đã được kiểm nghiệm!**")

if st.button("ARIMA + PC + DL", type="primary"):
    num_models = (len(ARIMA_PC_DL) - 1 + 5 + len(combines) *
                  (len(ARIMA_PC_DL) - 2 + 5))*len(file_names)
    with st.spinner("**Đang train và test tất cả ARIMA + PC + DL**"):
        model_number = 0
        progress_bar = st.progress(0)
        st.write(f"**Tổng số mô hình: {num_models}**")

        for file_name in file_names:
            metric_table = None
            arima_predict_set = None
            arima_fitted_set = None
            arima_test_time = None
            arima_train_parallel_time = None

            data = Dataset()
            data.data = pd.read_csv(data_path + file_name)
            data.preprocess_data()
            data.target_data = data.data[data.data.columns[0]]
            data.standardize_data(standardize_strategy, min_scale, max_scale)
            data.split_data(test_size, validate_size)

            # Auto-ARIMA model parameters
            trace_output = StringIO()
            with redirect_stdout(trace_output):
                model = auto_arima(
                    data.train_data, trace=True, suppress_warnings=True,
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
                time_run = parts[2].replace("Time=", "")

                trace_df.append((order, aic, time_run))
            columns = ['Order', 'AIC', 'Time']
            trace_df = pd.DataFrame(trace_df, columns=columns)

            summary = model.get_params()["order"]
            best_p, best_d, best_q = (
                int(summary[0]),
                int(summary[1]),
                int(summary[2]),
            )

            # Train và Test đơn lẻ một mô hình
            for model_name in ARIMA_PC_DL:
                st.write(model_name)
                if model_name != "Polynomial Classifier":
                    model_number += 1
                    progress_percent = int((model_number / num_models) * 100)
                    progress_bar.progress(progress_percent)

                if metric_table is not None:
                    temp_metric_table = metric_table
                else:
                    temp_metric_table = None

                if model_name == "ARIMA":
                    model = ARIMAModel(
                        data.train_data, data.test_data, best_p, best_d, best_q)

                    # Train
                    start_time = time()
                    model.train()
                    train_time = float("{:.2f}".format(time() - start_time))

                    # Test
                    start_time = time()
                    predict, actual = model.test()
                    test_time = float("{:.2f}".format(time() - start_time))

                    arima_predict_set = predict
                    arima_test_time = test_time

                    # Score
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                    metrics.update({"Thời gian train": train_time,
                                    "Thời gian test": test_time})
                    temp_metric_table = pd.DataFrame(
                        metrics, index=[model_name])
                    if temp_metric_table is not None:
                        merge = [metric_table, temp_metric_table]
                        metric_table = pd.concat(merge)

                elif model_name == "Polynomial Classifier":
                    for degree in range(1, degrees + 1):
                        model_number += 1
                        progress_percent = int(
                            (model_number / num_models) * 100)
                        progress_bar.progress(progress_percent)
                        

                        model = PolynomialClassifierModel(
                            data.train_data, data.test_data, d, t, degree, epoch, batch_size)

                        # Train
                        start_time = time()
                        model.train()
                        train_time = float(
                            "{:.2f}".format(time() - start_time))

                        # Test
                        start_time = time()
                        predict, actual = model.test()
                        test_time = float("{:.2f}".format(time() - start_time))

                        # Score
                        predict_inv = rescale_data(
                            data.target_data, predict, standardize_strategy)
                        actual_inv = rescale_data(
                            data.target_data, actual, standardize_strategy)
                        metrics = score(predict_inv, actual_inv)
                        metrics.update({"Thời gian train": train_time,
                                        "Thời gian test": test_time})
                        temp_metric_table = pd.DataFrame(
                            metrics, index=[f"{model_name} bậc {degree}"])
                        if temp_metric_table is not None:
                            merge = [metric_table, temp_metric_table]
                            metric_table = pd.concat(merge)

                else:
                    best_model_path_json = model_path + file_name + "/" + model_name + ".json"

                    with open(best_model_path_json, 'r') as json_file:
                        hyperparameters = json.load(json_file)

                    # Now you can access hyperparameters like units, epochs, batch_size, learning_rate
                    best_unit = hyperparameters['units']
                    best_epoch = hyperparameters['epochs']
                    best_batch_size = hyperparameters['batch_size']
                    best_learning_rate = hyperparameters['learning_rate']


                    # Prepare training data
                    X_train, y_train = prepare_data(data.train_data, d, t)
                    X_train_rbfnn = X_train.copy()
                    X_train = X_train.reshape(
                        X_train.shape[0], X_train.shape[1], 1)

                    # Prepare validating data
                    X_valid, y_valid = prepare_data(data.valid_data, d, t)
                    X_valid = X_valid.reshape(
                        X_valid.shape[0], X_valid.shape[1], 1)

                    if model_name == "RBFNN":
                        model = RBFNNModel(data.train_dl_data, data.test_data, data.valid_data,
                                           d, t, best_unit, best_epoch, best_batch_size)
                    else:
                        model = DeepLearningModel(model_name, data.train_dl_data, data.test_data, data.valid_data,
                                                d, t, best_unit, best_epoch, best_batch_size)

                    if model_name == "RBFNN":
                        hyper_model = model.create_model(
                            X_train_rbfnn, t, best_unit, best_learning_rate)
                    else:
                        hyper_model = model.create_model(
                            model_name, t, best_unit, best_learning_rate)
                        
                    best_model_path = model_path + file_name + "/" + model_name + ".keras"
                    mc = ModelCheckpoint(best_model_path, monitor='val_loss',
                                         mode='min', verbose=1, save_best_only=True),

                    # Train
                    start_time = time()

                    # Compile and fit the model
                    hyper_model.fit(X_train, y_train, epochs=best_epoch, batch_size=best_batch_size,
                                    callbacks=[mc], shuffle=False, validation_data=(X_valid, y_valid))
                    
                    train_time = float("{:.2f}".format(time() - start_time))
                    
                    if model_name == "RBFNN":
                        hyper_model = load_model(best_model_path, custom_objects={'RBFLayer': RBFLayer})
                    else:
                        hyper_model = load_model(best_model_path)


                    model.model = hyper_model

                    # Test
                    start_time = time()
                    predict, actual = model.test()
                    test_time = float("{:.2f}".format(
                        time() - start_time))

                    # Score
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                    metrics.update({"Thời gian train": train_time,
                                    "Thời gian test": test_time})
                    temp_metric_table = pd.DataFrame(
                        metrics, index=[model_name])
                    if temp_metric_table is not None:
                        merge = [metric_table, temp_metric_table]
                        metric_table = pd.concat(merge)

            # Train và Test mô hình kết hợp
            for model_name in ARIMA_PC_DL[1:]:
                for combine in combines:
                    st.write("ARIMA", model_name, combine)
                    if model_name != "Polynomial Classifier":
                        model_number += 1
                        progress_percent = int((model_number / num_models) * 100)
                        progress_bar.progress(progress_percent)

                    if metric_table is not None:
                        temp_metric_table = metric_table
                    else:
                        temp_metric_table = None

                    if model_name == "Polynomial Classifier":
                        for degree in range(1, degrees + 1):
                            st.write("Bậc", degree)
                            model_number += 1
                            progress_percent = int((model_number / num_models) * 100)
                            progress_bar.progress(progress_percent)
                            if metric_table is not None:
                                temp_metric_table = metric_table
                            else:
                                temp_metric_table = None

                            model = CombiningModel("ARIMA", model_name, combine, data.train_data, None, data.test_data, None, None,
                                                best_p, 1, degree, unit, epoch, batch_size,
                                                best_p, best_d, best_q, False, None, None, None,
                                                arima_predict_set, arima_fitted_set)
                            # Train
                            start_time = time()
                            model.train()
                            train_time = float("{:.2f}".format(time() - start_time))

                            if arima_fitted_set is None:
                                arima_fitted_set = model.arima_fitted_set
                            if combine == "Song song":
                                if arima_train_parallel_time is None:
                                    arima_train_parallel_time = train_time
                                else:
                                    train_time += arima_train_parallel_time

                            # Test
                            start_time = time()
                            predict, actual = model.test()
                            test_time = float("{:.2f}".format(
                                time() - start_time)) + arima_test_time

                            # Score
                            predict_inv = rescale_data(
                                data.target_data, predict, standardize_strategy)
                            actual_inv = rescale_data(
                                data.target_data, actual, standardize_strategy)
                            metrics = score(predict_inv, actual_inv)
                            metrics.update({"Thời gian train": train_time,
                                            "Thời gian test": test_time})
                            temp_metric_table = pd.DataFrame(
                                metrics, index=[f"ARIMA {combine} Polynomial Classifier bậc {degree}"])
                            if temp_metric_table is not None:
                                merge = [metric_table, temp_metric_table]
                                metric_table = pd.concat(merge)
                    else:
                        model = CombiningModel("ARIMA", model_name, combine, data.train_data, data.train_dl_data, data.test_data,
                                            data.valid_data, validate_size, best_p, 1, None, unit, epoch, batch_size,
                                            best_p, best_d, best_q, False, None, None, None,
                                            arima_predict_set, arima_fitted_set)

                    # Train
                    start_time = time()
                    model.train()
                    train_time = float("{:.2f}".format(time() - start_time))

                    if arima_fitted_set is None:
                        arima_fitted_set = model.arima_fitted_set
                    if combine == "Song song":
                        if arima_train_parallel_time is None:
                            arima_train_parallel_time = train_time
                        else:
                            train_time += arima_train_parallel_time

                    # Test
                    start_time = time()
                    predict, actual = model.test()
                    test_time = float("{:.2f}".format(
                        time() - start_time)) + arima_test_time

                    # Score
                    predict_inv = rescale_data(
                        data.target_data, predict, standardize_strategy)
                    actual_inv = rescale_data(
                        data.target_data, actual, standardize_strategy)
                    metrics = score(predict_inv, actual_inv)
                    metrics.update({"Thời gian train": train_time,
                                    "Thời gian test": test_time})
                    temp_metric_table = pd.DataFrame(
                        metrics, index=[f"ARIMA {combine} {model_name}"])
                    if temp_metric_table is not None:
                        merge = [metric_table, temp_metric_table]
                        metric_table = pd.concat(merge)

            params = ["dataset_name", "column_name", "d", "t",
                      "split", "train_size", "test_size", "validate_size",
                      "standardize", "p_arima", "d_arima", "q_arima"]
            values = [file_name, data.data.columns[0], d, t,
                      f"{100 - test_size}/{test_size}", data.train_data.size, data.test_data.size, data.valid_data.size,
                      standardize_strategy, best_p, best_d, best_q]
            meanings = ["Tên tập dữ liệu", "Tên cột dự đoán", "Số ngày dùng để dự đoán", "Số ngày để dự đoán",
                        "Tỉ lệ tập train/test", "Kích thước tập train", "Kích thước tập test", "Kích thước tập validate",
                        "Phương thức chuẩn hóa", "Số bậc tự hồi qui (AR)", "Số bậc sai phân (I)", "Số bậc trung bình trượt (MA)"]

            info_table = pd.DataFrame({
                "Tham số": params,
                "Giá trị": values,
                "Ý nghĩa": meanings
            })

            save_result(None, metric_table, info_table,
                        f'{file_name} ARIMA_PC_DL', trace_df)

        st.success("**Tất cả mô hình đã được kiểm nghiệm!**")

if st.button("Tìm Hyperparamerter và Lưu mô hình DL tốt nhất", type="primary"):
    num_models = len(DL) * len(file_names)
    with st.spinner("**Đang tìm tham số cho tất cả mô hình học sâu**"):
        model_number = 0
        progress_bar = st.progress(0)
        st.write(f"**Tổng số mô hình: {num_models}**")

        for file_name in file_names:
            data = Dataset()
            data.data = pd.read_csv(data_path + file_name)
            data.preprocess_data()
            data.target_data = data.data[data.data.columns[0]]
            data.standardize_data(standardize_strategy, min_scale, max_scale)
            data.split_data(test_size, validate_size)

            for model_name in DL:
                model_number += 1
                progress_percent = int((model_number / num_models) * 100)
                progress_bar.progress(progress_percent)

                if model_name == "RBFNN":
                    model = RBFNNModel(data.train_dl_data, data.test_data, data.valid_data,
                                       d, t, unit, epoch, batch_size)
                else:
                    model = DeepLearningModel(model_name, data.train_dl_data, data.test_data, data.valid_data,
                                            d, t, unit, epoch, batch_size)
                result = model.hyperparameter()
                trials_df = []
                for res in result:
                    params = res['params']
                    target = res['target']
                    row = {
                        "validation_loss": -target,
                        "batch_size": int(params["batch_size"]),
                        "epoch": int(params["epoch"]),
                        "learning_rate": params["learning_rate"],
                        "unit": int(params["unit"])
                    }
                    trials_df.append(row)

                trials_df = pd.DataFrame(trials_df)
                # Find row with lowest validation loss
                best_trial_index = trials_df['validation_loss'].idxmin(
                )

                best_parameter = trials_df.iloc[best_trial_index]

                best_unit = int(best_parameter["unit"])
                best_epoch = int(best_parameter["epoch"])
                best_batch_size = int(best_parameter["batch_size"])
                best_learning_rate = best_parameter["learning_rate"]

                hyperparameters = {
                    'units': best_unit,
                    'epochs': best_epoch,
                    'batch_size': best_batch_size,
                    'learning_rate': best_learning_rate
                }

                best_model_path_json = model_path + file_name + "/" + model_name + ".json"
                # Path to the directory
                directory = os.path.dirname(best_model_path_json)

                # Create the directory if it doesn't exist
                os.makedirs(directory, exist_ok=True)

                # Save hyperparameters to a JSON file
                with open(best_model_path_json, 'w') as json_file:
                    json.dump(hyperparameters, json_file)


                # Prepare training data
                X_train, y_train = prepare_data(data.train_dl_data, d, t)
                X_train_rbfnn = X_train.copy()
                X_train = X_train.reshape(
                    X_train.shape[0], X_train.shape[1], 1)

                # Prepare validating data
                X_valid, y_valid = prepare_data(data.valid_data, d, t)
                X_valid = X_valid.reshape(
                    X_valid.shape[0], X_valid.shape[1], 1)
                
                if model_name == "RBFNN":
                    hyper_model = model.create_model(
                        X_train_rbfnn, t, best_unit, best_learning_rate)
                else:
                    hyper_model = model.create_model(
                        model_name, t, best_unit, best_learning_rate)

                best_model_path = model_path + file_name + "/" + model_name + ".keras"
                mc = ModelCheckpoint(best_model_path, monitor='val_loss',
                                     mode='min', verbose=1, save_best_only=True),

                # Compile and fit the model
                hyper_model.fit(X_train, y_train, epochs=best_epoch, batch_size=best_batch_size,
                                callbacks=[mc], shuffle=False, validation_data=(X_valid, y_valid))
        st.success("**Tất cả mô hình đã được kiểm nghiệm!**")
