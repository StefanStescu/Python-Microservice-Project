import pandas as pd
from typing import List
from models import SleepRecord
import numpy as np
from pydantic import ValidationError

# DataLoader class to handle loading and preprocessing of the dataset
# This class encapsulates the logic for loading, cleaning, and validating sleep data
# It uses the SleepRecord Pydantic model to validate each record
# It also provides methods to load the CSV file, preprocess the data, and validate records
class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.raw_df = None
        self.cleaned_df = None

    def load_csv(self) -> pd.DataFrame:
        """Load data from raw CSV file."""
        try:
            df = pd.read_csv(self.filepath)
            df.columns = df.columns.str.replace(" ", "_", regex=False)
            self.raw_df = df
            return df
        except Exception as e:
            raise ValueError(f"Could not load CSV file: {e}")

#   Preprocess the dataset
    # This method cleans the dataset by filtering out invalid records and filling missing values
    # It also converts categorical variables to appropriate types and creates new derived columns
    # The method returns a cleaned DataFrame ready for analysis
    def preprocess(self) -> pd.DataFrame:
        """Clean and filter the dataset."""
        if self.raw_df is None:
            self.load_csv()

        assert self.raw_df is not None, "Data must be loaded before preprocessing."
        df = self.raw_df.copy()

        # Filtering like in original logic
        df = df[
            (df["Sleep_Duration"].between(5.0, 7.5)) &
            (df["Stress_Level"].between(2, 9)) &
            (df["Quality_of_Sleep"].between(3, 9))
        ]

        # Fill missing categorical values
        df["Sleep_Disorder"] = df["Sleep_Disorder"].fillna("None")

        # Categoricals
        df["Sleep_Disorder"] = pd.Categorical(
            df["Sleep_Disorder"], categories=["Insomnia", "None", "Sleep Apnea"]
        )

        # Create stress category
        df["Stress_Category"] = pd.cut(
            df["Stress_Level"], bins=[0, 3, 5, 9], labels=["Low", "Moderate", "High"]
        )

        self.cleaned_df = df
        return df
        
#   Validate records using the Pydantic model
    # This method iterates through each row of the cleaned DataFrame and validates it against the SleepRecord model
    # It returns a list of valid SleepRecord instances, or raises a ValidationError if any record is invalid
    def validate_records(self) -> List[SleepRecord]:
        """Validate each record using the Pydantic model."""
        if self.cleaned_df is None:
            self.preprocess()

        assert self.cleaned_df is not None, "Data must be preprocessed before validation."

        valid_records = []
        for _, row in self.cleaned_df.iterrows():
            try:
                record = SleepRecord.model_validate(row.to_dict())
                valid_records.append(record)
            except ValidationError as e:
                print(f"Validation error for row {row.name}: {e}")
                continue

        return valid_records
