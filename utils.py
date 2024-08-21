import os
import win32com.client
from typing import Tuple

import pandas as pd
import numpy as np
import streamlit as st
import openpyxl as ox
import xlwings as xl
import xlsxwriter as xw
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

output_path = "../report/output.xlsx"


@st.cache_data
def prepare_data(data: np.array, d: int, t: int) -> Tuple[np.array, np.array]:
    """
    Prepares the data for time series forecasting.

    Parameters:
    - data (np.array): The input data array.
    - d (int): The number of lag observations as input (default=0).
    - t (int): The number of future observations to predict (default=0).

    Returns:
    - Tuple[np.array, np.array]: A tuple containing the input data array (X) and the target data array (y).

    Notes:
    - If d is 0, the input data array is reshaped to a column vector and returned as both X and y.
    - For each observation in the input data array, d lag observations are used as input (X) and t future observations are used as target (y).
    - The input data array (X) is reshaped to have shape (n_samples, d), where n_samples is the number of observations.
    - The target data array (y) is reshaped to have shape (n_samples, t), where n_samples is the number of observations.

    Example:
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> d = 2
    >>> t = 1
    >>> X, y = prepare_data(data, d, t)
    >>> print(X)
    array([[1, 2],
        [2, 3],
        [3, 4]])
    >>> print(y)
    array([[3],
        [4],
        [5]])
    """
    X = []
    y = []

    if d == 0:
        data = np.array(data).reshape(-1).reshape(len(data), 1)
        return data, data

    X = [data[i:i+d] for i in range(len(data) - d - t + 1)]
    y = [data[i+d:i+d+t] for i in range(len(data) - d - t + 1)]

    X = np.array(X).reshape(len(X), d)
    y = np.array(y).reshape(len(y), t)
    return X, y


@st.cache_data
def score(predict: np.array, actual: np.array) -> dict:
    """
    Calculate various evaluation metrics for predicted and actual values.

    Parameters:
        predict (np.array): Array of predicted values.
        actual (np.array): Array of actual values.

    Returns:
        dict: A dictionary containing the following evaluation metrics:
            - MAE (Mean Absolute Error)
            - MSE (Mean Squared Error)
            - RMSE (Root Mean Squared Error)
            - MAPE (%) (Mean Absolute Percentage Error)
            - CV(RMSE) (Coefficient of Variation of RMSE)
    """
    mae = np.round(mean_absolute_error(actual, predict), 4)
    mse = np.round(mean_squared_error(actual, predict), 4)
    rmse = np.round(np.sqrt(mse), 4)
    cvrmse = np.round(rmse/np.mean(actual)*100, 4)

    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "CV(RMSE)(%)": cvrmse
    }
    return metrics


@st.cache_data
def rescale_data(df_target: pd.DataFrame, data: np.array, strategy: str) -> np.array:
    """
    Rescale the data using either Z-score or Min-Max scaling strategy.

    Parameters:
        df_target (pd.DataFrame): The target dataframe used to fit the scaler.
        data (np.array): The data to be rescaled.
        strategy (str): The rescaling strategy to be used. Options are 'Z-score' or 'Min-Max'.

    Returns:
        np.array: The rescaled data.

    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if strategy == 'Z-score':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    scaler.fit(df_target.values.reshape(-1, 1))
    data_inverse = scaler.inverse_transform(data)
    return data_inverse


@st.cache_data
def save_result(result_table: pd.DataFrame, metric_table: pd.DataFrame, info_table: pd.DataFrame, sheet_name: str = "Sheet1", trace_df = None) -> None:
    """
    Save the result of a model evaluation to an Excel file.

    Parameters:
    - result_table (pd.DataFrame): The table containing the model evaluation results.
    - metric_table (pd.DataFrame): The table containing the metric values.
    - info_table (pd.DataFrame): The table containing the model information.
    - sheet_name (str, optional): The name of the sheet to save the results to. Defaults to "Sheet1".

    Returns:
    None

    Raises:
    PermissionError: If there is a permission error while saving the file.
    NameError: If there is a name error while saving the file.
    """
    if os.path.exists('../report') is False:
        os.makedirs('../report')

    if os.path.exists(output_path) is False:
        ox.Workbook().save(output_path)

    temp_output_path = "temp_output.xlsx"
    if os.path.exists(temp_output_path) is True:
        os.remove(temp_output_path)

    try:
        with pd.ExcelWriter(temp_output_path, engine='xlsxwriter') as writer:
            worksheet = writer.book.add_worksheet(sheet_name)
            metric_table.to_excel(writer, sheet_name=sheet_name)
            if result_table is not None:
                info_table.to_excel(writer, sheet_name=sheet_name,
                                    index=False, startrow=4)
                result_table.to_excel(
                    writer, sheet_name=sheet_name, startcol=9)
                workbook = writer.book

                header_format = workbook.add_format(
                    {'bold': True, 'bg_color': 'green', 'color': 'white', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
                worksheet.conditional_format(
                    'A1:L1', {'type': 'cell', 'criteria': '!=', 'value': '""', 'format': header_format})

                border_format = workbook.add_format(
                    {'border': 1, 'align': 'center', 'valign': 'vcenter'})
                worksheet.conditional_format(
                    'A2:Z1000', {'type': 'no_blanks', 'format': border_format})

                empty_format = workbook.add_format(
                    {'text_wrap': True, 'align': 'center', 'valign': 'vcenter'})
                worksheet.write_blank('A1', None, empty_format)

                worksheet.merge_range(
                    'A4:C4', 'Thông tin mô hình', header_format)

                center_format = workbook.add_format(
                    {'align': 'center', 'valign': 'vcenter'})

                worksheet.set_column('B:B', None, center_format, {
                    'first_row': 5, 'last_row': 30})

            else:
                info_table.to_excel(writer, sheet_name=sheet_name,
                                    index=False, startcol=9)

                workbook = writer.book

                header_format = workbook.add_format(
                    {'bold': True, 'bg_color': 'green', 'color': 'white', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
                worksheet.conditional_format(
                    'A1:L1', {'type': 'cell', 'criteria': '!=', 'value': '""', 'format': header_format})
                worksheet.conditional_format(
                    'J15:L15', {'type': 'cell', 'criteria': '!=', 'value': '""', 'format': header_format})

                border_format = workbook.add_format(
                    {'border': 1, 'align': 'center', 'valign': 'vcenter'})
                worksheet.conditional_format(
                    'A2:Z1000', {'type': 'no_blanks', 'format': border_format})

                empty_format = workbook.add_format(
                    {'text_wrap': True, 'align': 'center', 'valign': 'vcenter'})
                worksheet.write_blank('A1', None, empty_format)

                worksheet.merge_range(
                    'J1:L1', 'Thông tin mô hình', header_format)

                center_format = workbook.add_format(
                    {'align': 'center', 'valign': 'vcenter'})

                worksheet.set_column('K:K', None, center_format, {
                    'first_row': 2, 'last_row': 30})

                # Create formats for conditional formatting
                green_format = workbook.add_format(
                    {'bg_color': 'green', 'color': 'white', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
                red_format = workbook.add_format(
                    {'bg_color': 'red', 'color': 'white', 'border': 1, 'align': 'center', 'valign': 'vcenter'})

                # Apply conditional formatting to each column
                for col_name in metric_table.columns:
                    col_data = metric_table[col_name]
                    min_value = col_data.min()
                    max_value = col_data.max()

                    # Apply green formatting to the cell with the smallest value
                    worksheet.conditional_format('B2:F66', {
                                                 'type': 'cell', 'criteria': '==', 'value': min_value, 'format': green_format})

                    # Apply red formatting to the cell with the largest value
                    worksheet.conditional_format('B2:F66', {
                                                 'type': 'cell', 'criteria': '==', 'value': max_value, 'format': red_format})

                if trace_df is not None:
                    trace_df.to_excel(writer, sheet_name=sheet_name,
                                                  index=False, startcol=9, startrow=14)
                    # Apply conditional formatting to each column
                    col_data = trace_df["AIC"]
                    min_value = col_data.min()

                    # Apply green formatting to the cell with the smallest value
                    worksheet.conditional_format('K16:K40', {
                                                'type': 'cell', 'criteria': '==', 'value': min_value, 'format': green_format})
                    
            worksheet.autofit()

        with xl.App(visible=False) as app:
            with xl.Book(temp_output_path) as source_wb, xl.Book(output_path) as target_wb:

                source_sheet = source_wb.sheets[sheet_name]
                try:
                    target_sheet = target_wb.sheets.add(sheet_name)
                except ValueError:
                    if target_wb.sheets[sheet_name] != target_wb.sheets[0]:
                        target_wb.sheets[sheet_name].api.Move(
                            target_wb.sheets[0].api)
                    target_sheet = target_wb.sheets[0]

                source_sheet.api.Copy(Before=target_sheet.api)

                target_wb.sheets[1].delete()

                target_wb.sheets[0].name = sheet_name
                target_wb.save()

        os.remove(temp_output_path)

        try:
            xll = win32com.client.gencache.EnsureDispatch("Excel.Application")
            xll.Quit()
        except Exception as e:
            st.write(f"An error occurred: {e}")

        st.success(
            f'**Kết quả đã được lưu trong report/output sheet {sheet_name}**')
    except (PermissionError, NameError) as error:
        st.write(error)


######################################### Hàm hỗ trợ #########################################
def reset_cache():
    """
    Resets the cache by setting all session state variables to None.

    This function is used to reset the cache in the Streamlit application. It sets all the session state variables related to data, predictions, actual values, models, model selections, training time, testing time, result table, metric table, and info table to None.

    Parameters:
        None

    Returns:
        None
    """
    st.session_state.data = None
    st.session_state.predict = None
    st.session_state.actual = None
    st.session_state.model = None
    st.session_state.model1_name = None
    st.session_state.model2_name = None
    st.session_state.train_time = None
    st.session_state.test_time = None
    st.session_state.result_table = None
    st.session_state.metric_table = None
    st.session_state.info_table = None


def clear_cache():
    """
    Clears the cache of the Streamlit session state.

    This function clears the cache of the Streamlit session state by resetting all the session state variables to their initial values. It is useful when you want to start fresh with the session state variables.

    Parameters:
        None

    Returns:
        None

    Example:
        clear_cache()

    Note:
        This function assumes that the session state variables have been initialized and assigned values before calling it.
    """
    data = st.session_state.data
    predict = st.session_state.predict
    actual = st.session_state.actual
    model = st.session_state.model
    model1_name = st.session_state.model1_name
    model2_name = st.session_state.model2_name
    train_time = st.session_state.train_time
    test_time = st.session_state.test_time
    result_table = st.session_state.result_table
    metric_table = st.session_state.metric_table
    info_table = st.session_state.info_table
    st.session_state.clear()
    st.session_state.data = data
    st.session_state.predict = predict
    st.session_state.actual = actual
    st.session_state.model = model
    st.session_state.model1_name = model1_name
    st.session_state.model2_name = model2_name
    st.session_state.train_time = train_time
    st.session_state.test_time = test_time
    st.session_state.result_table = result_table
    st.session_state.metric_table = metric_table
    st.session_state.info_table = info_table