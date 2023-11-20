import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import os

# Create a directory for the days if it doesn't exist
if not os.path.exists('days'):
    os.mkdir('days')

# Define the number of days
num_days = 7

# Number of points to choose for each latitude
num_points_per_latitude = 42

# List of dates from January 1, 2012, to January 7, 2012
dates = ["January 1, 2012", "January 2, 2012", "January 3, 2012", "January 4, 2012", "January 5, 2012", "January 6, 2012", "January 7, 2012"]

# Loop through the days and create frames
for day in range(1, num_days + 1):
    # Read zonal and meridional speed data for the current day
    with open(f'days/day{day}_z.xlsx', 'r', encoding='utf-8') as z_file:
        zonal_data = pd.read_csv(z_file, header=None, index_col=None)
    with open(f'days/day{day}_m.xlsx', 'r', encoding='utf-8') as m_file:
        meridional_data = pd.read_csv(m_file, header=None, index_col=None)

    # Convert DataFrames to NumPy arrays
    zonal_matrix = zonal_data.values
    meridional_matrix = meridional_data.values

    # Filter out elements with value -1.E+034
    filter_value = -1.E+034
    zonal_matrix_filtered = np.where(
        zonal_matrix == filter_value, np.nan, zonal_matrix)
    meridional_matrix_filtered = np.where(
        meridional_matrix == filter_value, np.nan, meridional_matrix)

    # Calculate the magnitude of the vector sum
    magnitude_matrix = np.sqrt(
        meridional_matrix_filtered ** 2 + zonal_matrix_filtered ** 2)

    # Size of the grid
    grid_size_x = 721  # Columns
    grid_size_y = 360  # Rows

    # Create a grid of points
    x = np.arange(grid_size_x)
    y = np.arange(grid_size_y)
    X, Y = np.meshgrid(x, y)

    # Calculate the vector sum direction (in radians)
    direction = np.arctan2(meridional_matrix_filtered, zonal_matrix_filtered)

    # Create a figure
    plt.figure(figsize=(20, 20))

    # Calculate latitude intervals
    lat_interval = grid_size_y // num_points_per_latitude

    # Initialize an empty list to store selected points
    selected_points = []

    # Loop through latitude lines and select points at uniform intervals
    for lat_line in range(0, grid_size_y, lat_interval):
        for i in range(0, grid_size_x, grid_size_x // num_points_per_latitude):
            selected_points.append((lat_line, i))

    # Convert the selected points into NumPy arrays
    selected_points = np.array(selected_points)
    selected_x = selected_points[:, 1]
    selected_y = selected_points[:, 0]

    # Use matplotlib.colormaps.get_cmap to get the colormap
    cmap = cm.nipy_spectral

    # Normalize the color map
    norm = Normalize(vmin=-0.6, vmax=22)

    # Calculate the arrow sizes based on wind velocity
    arrow_sizes = 0.5 + 0.05 * magnitude_matrix[selected_y, selected_x]

    # Create the colored points using a scatter plot with magnitude_matrix
    scatter = plt.scatter(X, Y, c=magnitude_matrix, cmap=cmap, norm=norm, s=1)

    # Draw gray dots on points with NaN values
    nan_x, nan_y = np.where(np.isnan(meridional_matrix_filtered))
    plt.scatter(nan_y, nan_x, c='white', s=1)  # Note the reversal of x and y

    # Set the aspect ratio to be equal
    plt.gca().set_aspect('equal')

    # Set the y-axis ticks and labels for specific latitudes
    lat_ticks = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324, 359]
    lat_labels = ['90°S', '70°S', '50°S', '30°S',
                '10°S', '0°', '10°N', '30°N', '50°N', '70°N', '90°N']
    plt.yticks(lat_ticks, lat_labels)

    lon_ticks = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720]
    lon_labels = ['0°', '30°E', '60°E', '90°E', '120°E', '150°E', '180°', '150°W', '120°W', '90°W', '60°W', '30°W', '0°']
    plt.xticks(lon_ticks, lon_labels)

    # Calculate the arrow magnitudes based on wind velocity
    arrow_magnitudes = 0.025 * magnitude_matrix[selected_y, selected_x]

    # Calculate the x and y components of the arrows based on direction
    selected_zonal = arrow_magnitudes * np.cos(direction[selected_y, selected_x])
    selected_meridional = arrow_magnitudes * np.sin(direction[selected_y, selected_x])

    # Create the quiver plot with arrow sizes based on wind velocity
    plt.quiver(selected_x, selected_y, selected_zonal, selected_meridional, angles='xy', scale_units='inches', color='black', pivot='middle', width=0.003, headwidth=2, headlength=4, scale=1)

    # Add streamlines to the plot
    plt.streamplot(x, y, zonal_matrix_filtered, meridional_matrix_filtered, color='white')

    cbar = plt.colorbar(scatter, orientation='horizontal')
    cbar.set_label('Magnitude')

    # Set the background color of the axes (the space outside the plot)
    plt.gca().set_facecolor('gray')

    # Add the date label to the frame
    date_label = dates[day - 1]
    plt.annotate(date_label, (0, 0), (20, 20), xycoords='axes fraction',
                textcoords='offset points', fontsize=12, color='red')

    # Save the plot as an image for this frame
    plt.savefig(f"images/January_{day}.png")
    plt.close()

# Create a GIF from the frames
images = [Image.open(f"images/January_{day}.png")
          for day in range(1, num_days + 1)]
images[0].save("images/GIF.gif", save_all=True,
               append_images=images[1:], loop=0, duration=2000)

print("GIF created!")
