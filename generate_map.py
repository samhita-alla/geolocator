import pandas as pd
import plotly
import plotly.express as px


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
