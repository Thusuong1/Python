from dataclasses import dataclass

import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@dataclass
class Dataset:
    """
    A class representing a dataset.

    Attributes:
        data (pd.DataFrame): The original dataset.
        target_data (pd.DataFrame): The target variable data.
        scaled_data (pd.DataFrame): The scaled target variable data.
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.

    Methods:
        load_data(uploaded_file: st.file_uploader) -> None:
            Loads the data from a CSV file.

        preprocess_data() -> None:
            Preprocesses the data by converting the first column to a datetime index,
            filling NaN values, and sorting the data.

        standardize_data(strategy: str, min_scale: int, max_scale: int) -> None:
            Standardizes the target variable data using either Z-score or Min-Max scaling.

        split_data(test_size: int) -> None:
            Splits the target variable data into training and testing datasets.
    """
    data: pd.DataFrame = None
    target_data: pd.DataFrame = None
    scaled_data: pd.DataFrame = None
    train_data: pd.DataFrame = None
    test_data: pd.DataFrame = None
    valid_data: pd.DataFrame = None

    def load_data(self, uploaded_file: st.file_uploader) -> None:
        self.data = pd.read_csv(uploaded_file)

    def preprocess_data(self) -> None:
        """
        Preprocesses the data by converting the first column to a datetime index,
        filling NaN values, and sorting the data.

        Parameters:
            None

        Returns:
            None
        """
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                try:
                    self.data[column] = pd.to_datetime(self.data[column])
                    self.data.set_index(column, inplace=True)
                except:
                    continue
        self.data.ffill(inplace=True)
        self.data.sort_index(inplace=True)

    def standardize_data(self, strategy: str, min_scale: int, max_scale: int) -> None:
        """
        Standardizes the target variable data using either Z-score or Min-Max scaling.

        Parameters:
            strategy (str): The scaling strategy to use. Options are 'Z-score' or 'Min-Max'.
            min_scale (int): The minimum value for Min-Max scaling. Ignored if strategy is 'Z-score'.
            max_scale (int): The maximum value for Min-Max scaling. Ignored if strategy is 'Z-score'.

        Returns:
            None
        """
        data = self.target_data.values.reshape(-1, 1)
        index = self.target_data.index
        name = self.target_data.name
        if strategy == 'Z-score':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler((min_scale, max_scale))

        scaled_data = scaler.fit_transform(data)

        self.scaled_data = pd.Series(
            scaled_data.flatten(), index=index, name=name)

    def split_data(self, test_size: int, validate_size: int = None) -> None:
        """
        Splits the target variable data into training and testing datasets.

        Parameters:
            test_size (int): The percentage of data to be used for testing. Should be between 0 and 100.
            validate_size (int, optional): The percentage of data to be used for validation. Should be between 0 and 100. Defaults to None.

        Returns:
            None
        """
        self.train_data, self.test_data = train_test_split(
            self.target_data if self.scaled_data is None else self.scaled_data,
            test_size=test_size / 100,
            shuffle=False
        )

        if validate_size is not None:
            self.valid_data, self.train_data = train_test_split(
                self.train_data,
                train_size=validate_size / 100,
                shuffle=False
            )
