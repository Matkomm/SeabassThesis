from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load fish data from your CSV file
fish_data = pd.read_csv(r'C:\Users\Mathiako\OneDrive - NTNU\Documents\Master Mathias\Master\Simulation_data\afternoon_test2.csv')

# Filter out the data for Fish 6
fish6_data = fish_data[fish_data['fish_id'] == 6]

# Create the figure
fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

# Add the fish's path as a line
fig.add_trace(go.Scatter3d(
    x=fish6_data['pos_x'],
    y=fish6_data['pos_y'],
    z=fish6_data['pos_z'],
    mode='lines',
    name='Fish 6 Path'
))

# Add a point that will represent the fish, starting at the first position
fish_marker = go.Scatter3d(
    x=[fish6_data['pos_x'].iloc[0]],
    y=[fish6_data['pos_y'].iloc[0]],
    z=[fish6_data['pos_z'].iloc[0]],
    mode='markers',
    name='Fish Position',
    marker=dict(size=5, color='red')
)
fig.add_trace(fish_marker)

# Update layout to add the slider
steps = []
for i in range(len(fish6_data)):
    step = dict(
        method='relayout',  # Use 'relayout' to update layout properties (e.g., marker position)
        args=[
            {'scene.marker.x': fish6_data['pos_x'].iloc[i]},  # New x coordinate for the fish position
            {'scene.marker.y': fish6_data['pos_y'].iloc[i]},  # New y coordinate for the fish position
            {'scene.marker.z': fish6_data['pos_z'].iloc[i]}   # New z coordinate for the fish position
        ],
        label=str(fish6_data['time_step'].iloc[i])  # The label for the step is set to the corresponding time step
    )
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time Step: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

# Plot the figure
fig.show()
