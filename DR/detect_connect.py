import pandas as pd
import numpy as np
import plotly.express as px
import collections
import os
import glob
from datetime import timedelta
import plotly.graph_objects as go
from typing import List, Dict, Tuple
from datetime import datetime


def read_data(date):
    # Extract year, month, and day from the date string
    year, month, day = date.split("-")

    # Initialize empty lists to store dataframes
    dfs_dayin = []
    dfs_input = []
    dfs_dayout = []
    dfs_output = []
    dfs_couples = []

    # Loop over both infrastructures
    for infra in ["apcn", "cloud"]:
        # Construct the path
        path = f"data/{infra}/{date}"

        # Read the data and append to the corresponding list
        dfs_dayin.append(pd.read_csv(path + "/DAILY_IN.csv", sep=";", header=None))
        dfs_input.append(pd.read_csv(path + "/INPUT.csv", sep=";", header=None))
        dfs_dayout.append(pd.read_csv(path + "/DAILY_OUT.csv", sep=";", header=None))
        dfs_output.append(pd.read_csv(path + "/OUTPUT.csv", sep=";", header=None))
        dfs_couples.append(pd.read_csv(path + "/couples.csv", sep=";", header=None))

    # Concatenate the dataframes
    df_dayin = pd.concat(dfs_dayin, ignore_index=True)
    df_input = pd.concat(dfs_input, ignore_index=True)
    df_dayout = pd.concat(dfs_dayout, ignore_index=True)
    df_output = pd.concat(dfs_output, ignore_index=True)
    df_couples = pd.concat(dfs_couples, ignore_index=True)
    
    return df_dayin, df_input, df_dayout, df_output, df_couples



## Read the dataframe again and add the header as the column names
def set_header(df, header):
    df.columns = header
    return df


# df_dayin, df_input, df_dayout, df_output, df_couples = read_data('cloud', 2023, 3, 27)

# # Set the header for each dataframe
# set_header(df_couples, ['id', 'router', 'interface', 'SAMPLES', 'infrastructure'])
# set_header(df_dayin, ['id', 'router', 'interface', 'DATE', 'MAXvalue', 'AVERAGE', 'PERCENTILE', 'CROSS', 'RATIOMAX', 
#                       'RATIOAVERAGE', 'RATIOPERCENTILE', 'SAMPLES', 'infrastructure'])
# set_header(df_input, ['id', 'router', 'interface', 'timestamp', 'value'])
# set_header(df_output, ['id', 'router', 'interface', 'timestamp', 'value'])


def get_unique_infrastruct(df):
    df_uni  = df.copy()
    unique_values = df_uni["infrastructure"].unique().tolist()
    return unique_values
# Assuming your DataFrame is named unmatched_df
# unique_infrastructures = get_unique_infrastruct(df_couples)


def get_subinterfaces_less_than_d_times(df: pd.DataFrame, df_couples: pd.DataFrame, n: int) -> list:
    """
    This function takes a DataFrame of timestamps and the number of days (d) and the number of missing timestamps per day (n).
    It returns a list of tuples, each containing the interface number, router name, the count of timestamps for that interface, and the infrastructure,
    for all interfaces that have a count less than d*(288-n).
    """
    # Group the DataFrame by router and interface, and count the number of timestamps for each interface
    timestamps_count = df.groupby(['router', 'interface'])['timestamp'].count()
    
    # Convert the Series to DataFrame and reset the index
    timestamps_count_df = timestamps_count.reset_index()

    # Merge timestamps_count_df with df_couples to get corresponding infrastructure for each (router, interface) pair
    merged_df = pd.merge(timestamps_count_df, df_couples, on=['router', 'interface'], how='left')
    
    # Calculate the minimum number of timestamps required for a interface
    min_timestamps = (288 - n)
    
    # Filter subinterfaces that have a count less than the minimum required
    subinterfaces_less_than_d_times = merged_df[merged_df['timestamp'] < min_timestamps]
    
    # Retrieve the router name, interface, count, and infrastructure for each interface
    fullinterfaces_info = list(subinterfaces_less_than_d_times[['router', 'interface', 'timestamp', 'infrastructure']].itertuples(index=False, name=None))
    
    return fullinterfaces_info

# Call the function with df_dayin (or df_output) and df_couples
# fullinterfac/es_info = get_subinterfaces_less_than_d_times(df_dayin, df_couples, 30)
# Print fist 5 elements of the list
# print(fullinterfaces_info[:5])



def visualize_subinterfaces(df: pd.DataFrame, fullinterfaces_info: list):
    """
    This function visualizes the interfaces by plotting line charts for each interface.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamps and values.
        fullinterfaces_info (list): List of tuples containing the interface information.

    """
    
    # Create an empty figure
    fig = go.Figure()

    # Iterate over the subinterfaces
    for router, interface, count, infrastructure in fullinterfaces_info:
        # Filter the DataFrame for the current interface
        sub_df = df[(df['router'] == router) & (df['interface'] == interface)].copy()

        # Convert the timestamp column to datetime type
        sub_df['timestamp'] = pd.to_datetime(sub_df['timestamp'])

        # Create a DataFrame with all possible timestamps for the interface
        timestamps = pd.date_range(start=sub_df['timestamp'].min(), end=sub_df['timestamp'].max(), freq='5min')
        fill_df = pd.DataFrame({'timestamp': timestamps})

        # Convert the timestamp column to datetime type
        fill_df['timestamp'] = pd.to_datetime(fill_df['timestamp'])

        # Merge the original DataFrame with the DataFrame of all timestamps
        merged_df = pd.merge(fill_df, sub_df, on='timestamp', how='left')

        # Add a line trace to the figure
        fig.add_trace(
            go.Scatter(
                x=merged_df['timestamp'],
                y=merged_df['value'],
                mode='lines',
                name=f"{router} {interface} ({infrastructure})"
            )
        )

    # Set the x-axis label and title
    fig.update_xaxes(title_text='Timestamp')
    # fig.update_layout(title_text='Missing Interface Visualization')

    # Show the plot
    # fig.show()

    return fig

# visualize_subinterfaces(df_dayin, fullinterfaces_info)


def get_dead_and_not_recovered_interfaces(df: pd.DataFrame, df_couples: pd.DataFrame, n: int, start_time: str, end_time: str) -> list:
    df = df.copy()

    # Convert df['timestamp'] to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Group the DataFrame by router and interface, and count the number of timestamps for each interface
    timestamps_count = df.groupby(['router', 'interface'])['timestamp'].count()
    # Calculate the minimum number of timestamps required for an interface
    min_timestamps = 288 - n
    # Filter subinterfaces that have a count less than the minimum required
    subinterfaces_dead = timestamps_count[timestamps_count < min_timestamps]

    start_time = datetime.strptime(start_time, '%H:%M').time()
    end_time = datetime.strptime(end_time, '%H:%M').time()

    # Get all interfaces
    all_interfaces = set(df[['router', 'interface']].itertuples(index=False, name=None))
    
    # Get interfaces in the last time range
    last_time_range_df = df[(df['timestamp'].dt.time >= start_time) & (df['timestamp'].dt.time <= end_time)]
    interfaces_in_last_time_range = set(last_time_range_df[['router', 'interface']].itertuples(index=False, name=None))

    # Identify interfaces that are dead and not present in the last time range
    subinterfaces_dead_and_not_recovered = all_interfaces.difference(interfaces_in_last_time_range)

    # Convert the Series to DataFrame and reset the index
    timestamps_count_df = timestamps_count.reset_index()

    # Merge timestamps_count_df with df_couples to get corresponding infrastructure for each (router, interface) pair
    merged_df = pd.merge(timestamps_count_df, df_couples, on=['router', 'interface'], how='left')

    # Retrieve the router name, interface, count, and infrastructure for each interface
    subinterfaces_info = [(router, interface, merged_df.loc[(merged_df['router']==router) & (merged_df['interface']==interface), 'timestamp'].values[0], merged_df.loc[(merged_df['router']==router) & (merged_df['interface']==interface), 'infrastructure'].values[0]) for (router, interface) in subinterfaces_dead_and_not_recovered if (router, interface) in subinterfaces_dead]

    return subinterfaces_info

def visualize_dead_and_not_recovered_interfaces(df: pd.DataFrame, dead_and_not_recovered_interfaces: list):
    """
    This function visualizes the dead and not recovered interfaces by creating line charts for each interface.

    Args:
        df (pd.DataFrame): The DataFrame containing the timestamps and values.
        dead_and_not_recovered_interfaces (list): List of tuples containing the dead and not recovered interface information.

    """
    # Create an empty figure
    fig = go.Figure()

    # Iterate over the dead and not recovered interfaces
    for router, interface, count, infrastructure in dead_and_not_recovered_interfaces:
        # Filter the DataFrame for the current interface
        sub_df = df[(df['router'] == router) & (df['interface'] == interface)].copy()

        # Convert the timestamp column to datetime type
        sub_df['timestamp'] = pd.to_datetime(sub_df['timestamp'])

        # Add a line trace to the figure
        fig.add_trace(
            go.Scatter(
                x=sub_df['timestamp'],
                y=sub_df['value'],
                mode='lines',
                name=f"{router} {interface} ({infrastructure})"
            )
        )

    # Set the x-axis label
    fig.update_xaxes(title_text='Timestamp')

    # Show the plot
    # fig.show()

    return fig



# start_time = '23:00'
# end_time = '23:55'
# n = 30

# dead_and_not_recovered_interfaces = get_dead_and_not_recovered_interfaces(df_input, df_couples, n, start_time, end_time)
# if len(dead_and_not_recovered_interfaces) > 0:
#     for (router, interface, count, infrastructure) in dead_and_not_recovered_interfaces:
#         print(f"Interface: {interface} on router: {router} with infrastructure: {infrastructure} has {count} timestamps and did not recover.")
# else:
#     print("No interfaces without recovery found.")

# fig = visualize_dead_and_not_recovered_interfaces(df_input, dead_and_not_recovered_interfaces)

# # Display the figure
# fig.show()



## read the interface/device from df_couples and df_input and report the unmatched interface/device
## the unmatched interface/device: the interface/device in df_couples but not in df_input

def get_unmatched_interfaces(df_couples: pd.DataFrame, df_input: pd.DataFrame):
    """
    This function reads the ('interface', 'router', 'infrastructure') from df_couples and df_input
    and reports the unmatched ('interface', 'router', 'infrastructure').

    Args:
        df_couples (pd.DataFrame): DataFrame containing the ('interface', 'router', 'infrastructure') from df_couples.
        df_input (pd.DataFrame): DataFrame containing the ('interface', 'router') from df_input.

    """
    # Get the unique ('interface', 'router', 'infrastructure') pairs from df_couples
    couples = set(zip(df_couples['router'], df_couples['interface']))

    # Get the unique ('interface', 'router') pairs from df_input
    inputs = set(zip(df_input['router'], df_input['interface']))

    # Find the unmatched ('interface', 'router', 'infrastructure') pairs
    unmatched = couples - inputs

    # Convert the unmatched pairs to a DataFrame
    unmatched_df = pd.DataFrame(unmatched, columns=['router', 'interface'])

    # Print the unmatched ('interface', 'router', 'infrastructure') pairs
    # print("The totally missing (interface, router, infrastructure) are:")
    # for interface, router, infrastructure in unmatched:
    #     print(f"{interface} {router} ({infrastructure})")

    return unmatched_df


# Get the unmatched interfaces
# unmatched_interfaces = get_unmatched_interfaces(df_couples, df_input)
# print(unmatched_interfaces)