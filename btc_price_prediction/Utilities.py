import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def plot_series(time, series, format="-", start=0, end=None, legend=None):
    """
    Visualizes time series data

    Args:
      time (array of int) - contains the time steps
      series (array of int or tuple) - contains the measurements for each time step
      format - line style when plotting the graph
      start - first time step to plot
      end - last time step to plot
      legend - legend labels for each series (if series is a tuple)
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))
    
    if type(series) is tuple:
        for series_num, label in zip(series, legend):
            # Plot the time series data
            plt.plot(time[start:end], series_num[start:end], format, label=label)
    else:
        # Plot the time series data
        plt.plot(time[start:end], series[start:end], format, label=legend)

    # Label the x-axis
    plt.xlabel("Time")
    plt.xticks(rotation=45)

    # Label the y-axis
    plt.ylabel("Value")

    # Overlay a grid on the graph
    plt.grid(True)

    # Add legend
    if legend is not None:
        plt.legend()

    # Draw the graph on screen
    plt.show()

    
def moving_average_forecast(series, window_size):
    """Generates a moving average forecast

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to compute the average for

    Returns:
      forecast (array of float) - the moving average forecast
    """

    # Initialize a list
    forecast = []
    
    # Compute the moving average based on the window size
    for time in range(len(series) - window_size):
      forecast.append(series[time:time + window_size].mean())
    
    # Convert to a numpy array
    forecast = np.array(forecast)

    return forecast   

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset window
    
    Args:
        series (array of float): contains the values of the series
        window_size (int): the number of time steps to average
        batch_size (int): the batch size
        shuffle_buffer (int): buffer size to use for the shuffle method
        
    Returns:
        dataset (TF format): TF Dataset contains time windows
    """
    
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    
    # Flatten the windows by putting its elements in a single batch. Prepare the windows to be tensors instead of the Dataset structure
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Group into features and labels 
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset