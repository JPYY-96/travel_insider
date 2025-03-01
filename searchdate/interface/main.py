import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from searchdate.params import *
from searchdate.ml_logic.data import get_data_with_cache, clean_data, load_data_to_bq
from searchdate.ml_logic.model import initialize_model, compile_model, train_model, evaluate_model
from searchdate.ml_logic.preprocessor import preprocess_features
from searchdate.ml_logic.registry import load_model, save_model, save_results
from searchdate.ml_logic.registry import mlflow_run, mlflow_transition_model

def preprocess() -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)


    query = f"""
        SELECT *
        FROM `{GCP_PROJECT}`.{BQ_DATASET}
    """



    # Retrieve data using `get_data_with_cache`
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{DATA_SIZE}.csv")
    data_query = get_data_with_cache(
        query=query,
        gcp_project=GCP_PROJECT,
        cache_path=data_query_cache_path,
        data_has_header=True
    )


    X = data_query.drop("totalFare", axis=1)
    y = data_query[["totalFare"]]

    X_processed = preprocess_features(X)
