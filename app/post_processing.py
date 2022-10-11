import logging
from typing import Tuple

import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def dbscan_clustering(lats_and_longs: List[List[float]]) -> np.ndarray:
    # DBSCAN clustering algorithm to cluster lats and longs
    # Why DBSCAN? -- robust to outliers; needn't specify the number of clusters; density-based
    lats_and_longs_standardized = StandardScaler().fit_transform(lats_and_longs)
    db = DBSCAN(eps=0.5, min_samples=3)
    db_fit = db.fit(lats_and_longs_standardized)
    labels = db_fit.labels_
    logging.info(f"DBSCAN cluster labels: {labels}")

    return labels


def plot_clusters(labels, lats_and_longs_standardized):
    # plot DBSCAN clusters
    plt.scatter(
        lats_and_longs_standardized[:, 0],
        lats_and_longs_standardized[:, 1],
        c=labels,
        cmap="Paired",
    )


def data_engineering(inference_df: pd.DataFrame) -> List[List[float]]:
    logging.info(f"Inference DF: {inference_df.head()}")

    inference_df_lats_longs = inference_df[["pred_lat", "pred_lng"]]
    logging.info(f"Inference Lats & Longs only DF: {inference_df_lats_longs}")

    lats_and_longs = inference_df_lats_longs.values.tolist()
    return lats_and_longs


def generate_prediction_helper(
    lats_and_longs: List[List[float]],
) -> Tuple[str, float, float]:
    labels = dbscan_clustering(lats_and_longs=lats_and_longs)

    # find the dense cluster
    dense_cluster_label = max(set(labels), key=list(labels).count)
    logging.info(f"Dense cluster label: {dense_cluster_label}")

    # get data labels belonging to the dense cluster
    indices = np.where(labels == dense_cluster_label)[0]
    dense_cluster_data = list(map(lats_and_longs.__getitem__, indices))
    logging.info(f"Dense cluster data: {dense_cluster_data}")

    # fetch lat and long mean
    lat_long_array = np.mean(np.array(dense_cluster_data, dtype=float), axis=0)
    latitude, longitude = lat_long_array[0], lat_long_array[1]
    logging.info(f"Latitude: {latitude}, Longitutde: {longitude}")

    # get location
    location = get_location(latitude=latitude, longitude=longitude)

    return location, latitude, longitude


def generate_prediction(
    inference_df: pd.DataFrame,
) -> str:
    lats_and_longs = data_engineering(inference_df=inference_df)
    location, *_ = generate_prediction_helper(lats_and_longs=lats_and_longs)
    return location
"""


def get_location(latitude: float, longitude: float) -> str:
    geolocator = Nominatim(user_agent="geolocater")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(f"{latitude},{longitude}", language="en")
    return location.address


def generate_prediction_logit(
    inference_df: pd.DataFrame,
) -> Tuple[str, float, float]:
    # get the most confident prediction (highest pred_logit)
    best_pred = (
        inference_df.sort_values(by=["pred_logit"], ascending=False)
        .reset_index()
        .loc[0]
    )

    latitude, longitude = best_pred["pred_lat"], best_pred["pred_lng"]
    logging.info(f"Latitude: {latitude}, Longitutde: {longitude}")

    # get location
    try:
        location = get_location(latitude=latitude, longitude=longitude)
    except Exception:
        return None, None, None

    return location, latitude, longitude
