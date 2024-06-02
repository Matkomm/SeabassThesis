import pandas as pd
import plotly.graph_objects as go

# Load the new data format
fish_data_new = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\morning_final2.csv')

# Check the unique number of fish (assuming each fish has a unique identifier if provided)
if 'fish_id' in fish_data_new.columns:
    unique_fish = fish_data_new['fish_id'].nunique()
    print(f"Unique fish count: {unique_fish}")
else:
    print("Fish ID column not found. Assuming 1000 fish as mentioned.")

# Group by time and depth to understand the data distribution
distribution = fish_data_new.groupby(['time_step', 'pos_z']).size().reset_index(name='counts')
print(distribution.head(10))

# Define the updated function
def create_2d_histogram(cluster_df, time_column: str = "time_step", depth_column: str = "pos_z", reverse_axis: bool = True):
    # Convert time steps to hours and minutes
    start_time = pd.Timestamp('2023-01-01 06:00:00')  # Starting at 06:00
    cluster_df['time_converted'] = start_time + pd.to_timedelta(cluster_df[time_column], unit='s')
    
    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=cluster_df['time_converted'],
        y=cluster_df[depth_column],
        colorscale='Blues',
        reversescale=True,
        nbinsx=64,
        nbinsy=48,
        xaxis='x',
        yaxis='y',
        colorbar={"title": 'Counts'}
    ))
    fig.add_hline(y=2, line_dash='dash', line_color='Red', line_width=4)
    if reverse_axis:
        fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Depth in m",
        showlegend=True,
        autosize=False, # Disable autosizing
        width=1200, # Set the width of the figure
        height=700 # Set the height of the figure to make it more square
    )
    fig.update_xaxes(tickformat='%H:%M', dtick=300000*2)  # Set x-axis to show time in HH:MM:SS format
    fig.update_annotations(font_size=30)
    fig.update_layout(font_size=30)
    fig.show()

# Call the function with your data
create_2d_histogram(fish_data_new)
