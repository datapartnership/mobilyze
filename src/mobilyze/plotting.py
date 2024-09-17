import datetime

import colorcet as cc
import datashader
import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from bokeh.models import (
    ColumnDataSource,
    HoverTool,
    Legend,
    Range1d,
    TabPanel,
    Tabs,
    Title,
)
from bokeh.plotting import figure
from dask.dataframe import DataFrame
from holoviews.element.tiles import CartoDark
from holoviews.operation.datashader import rasterize

COLORS = [
    "#4E79A7",  # Blue
    "#F28E2B",  # Orange
    "#E15759",  # Red
    "#76B7B2",  # Teal
    "#59A14F",  # Green
    "#EDC948",  # Yellow
    "#B07AA1",  # Purple
    "#FF9DA7",  # Pink
    "#9C755F",  # Brown
    "#BAB0AC",  # Gray
    "#7C7C7C",  # Dark gray
    "#6B4C9A",  # Violet
    "#D55E00",  # Orange-red
    "#CC61B0",  # Magenta
    "#0072B2",  # Bright blue
    "#329262",  # Peacock green
    "#9E5B5A",  # Brick red
    "#636363",  # Medium gray
    "#CD9C00",  # Gold
    "#5D69B1",  # Medium blue
]
width = 750  # default for bokeh plots
height = 600  # default for bokeh plots


def plot_spatial_distribution(ddf: DataFrame) -> hv.Overlay:
    """
    Plots the spatial distribution of mobility data points using coordinates. The plot is rendered using `datashader` for
    efficient visualization of large datasets and overlayed with map tiles.

    Parameters:
    -----------
    ddf : dask.dataframe.DataFrame
        A Dask DataFrame containing the 'longitude' and 'latitude' columns
        representing the geographic coordinates of mobility data points.

    Returns:
    --------
    hv.Overlay
        A Holoviews overlay consisting of the CartoDark map tiles and the
        spatial distribution of the mobility data points.
    """
    hv.extension("bokeh")

    ddf["latitude"] = ddf["latitude"].astype("float")  # type: ignore
    ddf["longitude"] = ddf["longitude"].astype("float")  # type: ignore
    ddf["horizontal_accuracy"] = ddf["horizontal_accuracy"].astype("float")  # type: ignore

    x, y = datashader.utils.lnglat_to_meters(ddf["longitude"], ddf["latitude"])  # type: ignore
    points = hv.Points(ddf.assign(x=x, y=y), ["x", "y"])
    points = rasterize(points).opts(  # type: ignore
        tools=["hover"],
        colorbar=True,
        colorbar_position="bottom",
        cmap=cc.fire,
        cnorm="eq_hist",
    )

    tiles = CartoDark().opts(
        title="Mobility Data Spatial Distribution",
        alpha=0.75,
        width=width,
        height=height,
        bgcolor="black",
    )

    return tiles * points


def plot_temporal_distribution(ddf: DataFrame):
    """
    Plots the temporal distribution of mobility data points.

    Parameters:
    -----------
    ddf : dask.dataframe.DataFrame
        A Dask DataFrame containing the 'longitude' and 'latitude' columns
        representing the geographic coordinates of mobility data points.
    """
    count = (
        ddf.groupby(["country", "date"], observed=False)["uid"]
        .count()
        .reset_index()
        .rename(columns={"uid": "count"})
        .compute()
        .pivot_table(values="count", index="date", columns="country", observed=True)
    )
    count.index = pd.to_datetime(count.index)

    p = figure(
        x_axis_type="datetime",
        y_axis_type="log",
        x_axis_label="Date",
        y_axis_label="Pings",
        title="Mobility Data Temporal Distribution",
        width=width,
    )
    source = ColumnDataSource(count)

    # Loop through the country columns and add a line for each one
    for i, country in enumerate(sorted(count.columns)):
        p.line(
            x="date",
            y=country,
            source=source,
            legend_label=country,
            line_width=2,
            color=COLORS[i],
        )

    # Configure the plot
    p.add_layout(p.legend[0], "right")
    p.legend.click_policy = "hide"
    p.title.text_font_size = "16pt"  # type: ignore

    return p


def plot_activity(activity: pd.DataFrame, variable="z_score", freq="D"):
    """
    Plots activity trends over time based on a specified variable and frequency,
    grouping by administrative divisions (shapeGroup) and shape names (shapeName).

    The function generates interactive line plots for different groups of activity
    data, with options to visualize data trends, including legends, tooltips,
    and zooming tools. It returns these plots in a tabbed layout for comparison
    across different groups.

    Parameters
    ----------
    activity : pd.DataFrame
        A pandas DataFrame containing activity data with at least the following columns:
        'date' (datetime), 'shapeGroup' (categorical), 'shapeName' (categorical), and
        the `variable` (e.g., "z_score") to plot.

    variable : str, optional
        The column in the DataFrame to be used for plotting. By default, 'z_score'.

    freq : str, optional
        The frequency at which to group the data. Must be a valid pandas offset alias.
        For example, 'W' for weekly, 'M' for monthly. By default, 'D' (daily).

    Returns
    -------
    Tabs
        A Bokeh Tabs object containing one tab for each `shapeGroup`, with interactive
        line plots showing the activity trends for each `shapeName` within the group.

    Notes
    -----
    - The function groups the data by 'date', 'shapeGroup', and 'shapeName', calculates
      the mean for each group, and then pivots the data for plotting.
    - The y-axis range is set dynamically based on the minimum and maximum values of
      the `variable`.
    - The plot includes various Bokeh interactive tools such as pan, zoom, and hover
      with tooltips displaying the date and the corresponding value.
    - Tabs are used to display separate plots for each group (`shapeGroup`).
    """
    tabs = []

    activity = (
        activity.groupby(["date", "shapeGroup", "shapeName"])[variable]
        .mean()
        .to_frame()
        .reset_index()
    )
    min_y = np.floor(activity[variable].min())
    max_y = np.ceil(activity[variable].max())

    for group in sorted(
        activity[activity["shapeGroup"].notnull()]["shapeGroup"].unique()
    ):
        data = (
            activity[activity["shapeGroup"] == group]
            .groupby(["date", "shapeName"])[variable]
            .mean()
            .to_frame()
        )
        data = data.pivot_table(
            values=[variable], index=["date"], columns=["shapeName"]
        )
        data.columns = [x[1] for x in data.columns]
        data = data.groupby(pd.Grouper(freq=freq)).mean()

        p = figure(
            title=f"Activity Trends: {variable}",
            width=width,
            height=height,
            x_axis_label="Date",
            x_axis_type="datetime",
            y_axis_label=f" {variable} (based on device density)",
            tools="pan,wheel_zoom,box_zoom,reset,save,box_select",
        )
        p.y_range = Range1d(min_y, max_y, bounds=(min_y, None))
        p.add_layout(
            Title(
                text="",
                text_font_size="12pt",
                text_font_style="italic",
            ),
            "above",
        )

        p.add_layout(
            Title(
                text="Activity trends (based on device density) for each time window and each administrative division",
                text_font_size="12pt",
                text_font_style="italic",
            ),
            "above",
        )
        p.add_layout(
            Title(
                text=f"Source: Veraset Movement. Creation date: {datetime.datetime.today().strftime('%d %B %Y')}. Feedback: datalab@worldbank.org.",
                text_font_size="10pt",
                text_font_style="italic",
            ),
            "below",
        )
        p.add_layout(Legend(), "right")
        p.add_tools(
            HoverTool(
                tooltips="Date: @x{%F}, Value: @y{0.0}",
                formatters={"@x": "datetime"},
            )
        )
        renderers = []
        for column, color in zip(data.columns, COLORS):
            r = p.line(
                data.index,
                data[column],
                legend_label=column,
                line_color=color,
                line_width=2,
            )
            r.visible = False
            renderers.append(r)

        # Select first to be visible
        renderers[0].visible = True

        p.legend.location = "bottom_left"
        p.legend.click_policy = "hide"
        p.title.text_font_size = "16pt"
        p.sizing_mode = "scale_both"

        tabs.append(
            TabPanel(
                child=p,
                title=group,
            )
        )

    # Return tabs in a layout
    return Tabs(tabs=tabs, sizing_mode="scale_both")


def plot_visits(data, title="Points of Interest Visit Trends"):
    """
    Creates a plot showing the number of visits to OpenStreetMap (OSM) points of interest (POI) over time.

    Parameters:
    ----------
    data : pandas.DataFrame
        DataFrame containing visit counts with dates as index and POI categories as columns.
    title : str, optional
        Title of the plot (default is "Points of Interest Visit Trends").
     Returns:
    -------
    bokeh.plotting.Figure
        A Bokeh figure object with the plotted visit data.
    """
    p = figure(
        title=title,
        width=width,
        height=height,
        x_axis_label="Date",
        x_axis_type="datetime",
        y_axis_label="Visits",
        y_axis_type="log",
        y_range=(0.9, 10 ** np.ceil(np.log10(np.max(data)))),
        tools="pan,wheel_zoom,box_zoom,reset,save,box_select",
    )
    p.add_layout(
        Title(
            text="Visits to Points of Interest Over Time",
            text_font_size="12pt",
            text_font_style="italic",
        ),
        "above",
    )
    p.add_layout(
        Title(
            text=f"Source: Veraset Movement. Creation date: {datetime.datetime.today().strftime('%d %B %Y')}. Feedback: datalab@worldbank.org.",
            text_font_size="10pt",
            text_font_style="italic",
        ),
        "below",
    )
    p.add_layout(Legend(), "right")

    # plot lines
    for column, color in zip(data.columns, COLORS):
        p.line(
            data.index[:-1],  # ignore last date
            data[column][:-1],
            legend_label=column,
            line_color=color,
            line_width=2,
        )
    p.add_tools(
        HoverTool(
            tooltips=[("Date", "@x{%F}"), ("Value", "@y{0.0}")],
            formatters={"@x": "datetime"},
        )
    )
    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    p.title.text_font_size = "16pt"
    # p.sizing_mode = "scale_width"

    return p


def plot_visits_by_group(df, group):
    tabs = []
    df = df[df["shapeGroup"] == group]

    for name in df["shapeName"].unique():
        data = (
            df[df["shapeName"] == name]
            .pivot_table("count", index=["date"], columns=["category"], aggfunc="sum")
            .resample("W")
            .sum()
        )
        p = plot_visits(
            data,
            title=f"Trends in Visits to Points of Interest in {name}",
        )
        tabs.append(
            TabPanel(
                child=p,
                title=name,
            )
        )
    return Tabs(tabs=tabs, sizing_mode="scale_both")


def plot_share_by_quantile(data, num_quantiles=10):
    """
    Plot the share of the total for each quantile. This function divides the input data into specified quantiles and plots
    the proportion of the total sum that each quantile represents.

    Parameters
    ----------
    data : array-like
        Input data to be divided into quantiles. Can be a list, NumPy array, or pandas Series.
    num_quantiles : int, optional
        The number of quantiles to divide the data into. Default is 10.

    Returns
    -------
    None
        Displays a bar chart showing the share of the total sum for each quantile
    """
    # Convert data to a NumPy array and sort it
    data = np.sort(np.array(data))

    # Calculate the total sum of the data
    total_sum = np.sum(data)

    # Number of data points per quantile
    points_per_quantile = len(data) // num_quantiles

    # Calculate the share of total for each quantile
    shares = []
    for i in range(num_quantiles):
        start_index = i * points_per_quantile
        end_index = (i + 1) * points_per_quantile
        quantile_sum = np.sum(data[start_index:end_index])
        shares.append(quantile_sum / total_sum)

    # Create quantile labels
    labels = [f"Quantile {i + 1}" for i in range(num_quantiles)]

    # Plot the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(labels, shares, color="orange")

    # Add labels and title
    plt.xlabel("Quantile")
    plt.ylabel("Share of Total")
    plt.title("Share of Total by Quantile")
    plt.ylim(0, 1)

    # Display the values on top of the bars
    for i, share in enumerate(shares):
        plt.text(i, share + 0.02, f"{share:.2%}", ha="center", va="bottom")

    # Show the plot
    plt.show()


def plot_boxplot(data, freq="W-SUN"):
    data = (
        data.groupby(["uid", pd.Grouper(key="date", freq=freq)])
        .agg(count=("datetime", "count"))
        .pivot_table(index="uid", columns=["date"], values="count")
    )
    # Convert the date index to string for better plot labeling
    # data.columns = data.columns.strftime("%Y-%m-%d")

    fig = px.box(
        data.unstack().reset_index(name="Values"),
        x="date",
        y="Values",
        title="Time Series of Boxplots",
        labels={"date": "Date", "Values": "Values"},
        log_y=True,
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )
    fig.show()


def plot_gini_curve(income_distribution):
    """
    Plots the Lorenz curve and calculates the Gini coefficient.
    """
    # Sort income distribution in ascending order
    income_sorted = np.sort(income_distribution)

    # Get the cumulative income as a percentage of total
    cumulative_income = np.cumsum(income_sorted)
    cumulative_income = cumulative_income / cumulative_income[-1]  # normalize

    # Get cumulative population percentage
    population_percentage = np.arange(1, len(income_distribution) + 1) / len(
        income_distribution
    )

    # Plot Lorenz curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        population_percentage, cumulative_income, label="Lorenz Curve", color="blue"
    )

    # Plot line of equality
    plt.plot([0, 1], [0, 1], label="Line of Equality", linestyle="--", color="red")

    plt.fill_between(
        population_percentage,
        cumulative_income,
        population_percentage,
        color="lightblue",
        alpha=0.5,
    )

    plt.title("Lorenz Curve with Gini Coefficient")
    plt.xlabel("Cumulative Share of Population")
    plt.ylabel("Cumulative Share of Total")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Calculate the Gini coefficient
    # Area under the Lorenz curve using the trapezoidal rule
    area_under_lorenz = np.trapz(cumulative_income, population_percentage)

    # Gini is 1 - 2 * area under Lorenz curve
    gini_coefficient = 1 - 2 * area_under_lorenz

    plt.text(
        0.6,
        0.4,
        f"Gini Coefficient: {gini_coefficient:.2f}",
        fontsize=12,
        color="black",
    )

    plt.show()

    return gini_coefficient
