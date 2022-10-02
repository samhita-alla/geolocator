import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim
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
    # not being called currently!
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


def get_location(latitude: float, longitude: float) -> str:
    geolocator = Nominatim(user_agent="geolocater")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(f"{latitude},{longitude}", language="en")
    return location.address


def get_plotly_graph(
    latitude: float, longitude: float, location: str
) -> plotly.graph_objects.Figure:
    lat_long_data = [[latitude, longitude, location]]
    map_df = pd.DataFrame(lat_long_data, columns=["latitude", "longitude", "location"])

    px.set_mapbox_access_token(
        "pk.eyJ1Ijoic2FtaGl0YS1hbGxhIiwiYSI6ImNsOGgwZ3lyajB0NWczb3F4cHU4dHhocmcifQ.gl4lARnWScZcHJHtXClrLg"
    )
    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        hover_name="location",
        color_discrete_sequence=["fuchsia"],
        zoom=5,
        height=300,
    )
    fig.update_layout(mapbox_style="dark")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


def generate_prediction(
    inference_df: pd.DataFrame,
) -> str:
    # data engineering
    lats_and_longs = data_engineering(inference_df=inference_df)

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

    return location
