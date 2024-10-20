import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Read the CSV files
interpolated_df = pd.read_csv('interpolated.csv')
normal_df = pd.read_csv('normal.csv')

# Extract data for the second data series from interpolated.csv
x2 = interpolated_df['x2']
y2 = interpolated_df['y2']

# Extract x1 and y1 from normal.csv
x1 = normal_df.index 
y1 = normal_df['y1']

# Create a DataFrame for normal.csv data
normal_data = pd.DataFrame({'x1': x1, 'y1': y1})

# Sort the data by x1 to ensure proper alignment
normal_data.sort_values('x1', inplace=True)

# Use forward fill to handle missing x1 values
normal_data['x1_filled'] = normal_data['x1'].fillna(method='ffill')

# Reindex y1 to x2 using forward fill
# First, set x1_filled as the index
normal_data.set_index('x1_filled', inplace=True)

# Reindex to x2, forward filling missing y1 values
y1_aligned = normal_data['y1'].reindex(x2, method='ffill')

# Handle any remaining NaN values (e.g., at the beginning)
y1_aligned.fillna(method='bfill', inplace=True)

# Create a figure
fig = go.Figure()

# Add the first data series (y1 aligned to x2)
fig.add_trace(
    go.Scatter(
        x=x2,
        y=y1_aligned,
        mode='lines',
        line=dict(color='blue', width=4),  # Increased line width
        marker=dict(symbol='circle', color='blue'),
        name='Data Series 1 (y1 vs x2)'
    )
)

# Add the second data series
fig.add_trace(
    go.Scatter(
        x=x2,
        y=y2,
        mode='markers',
        line=dict(color='red', width=0.1),
        marker=dict(color='red', size=1),
        name='Data Series 2 (y2 vs x2)'
    )
)

# Update layout
fig.update_layout(
    title='Plot of Two Data Series from normal.csv and interpolated.csv',
    xaxis=dict(
        title='X2 Axis (x2 from interpolated.csv)'
    ),
    yaxis=dict(
        title='Y Axis (common for y1 and y2)'
    ),
    legend=dict(
        x=0,
        y=1.15,
        orientation='h'
    ),
    margin=dict(l=40, r=40, t=80, b=40)
)

# Show gridlines for better readability
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Display the plot
fig.show()
