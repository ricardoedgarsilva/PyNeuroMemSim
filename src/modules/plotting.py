from modules.dependencies import *

def plot_mse_hist(config: dict):
    """
    Reads the MSE data from a CSV file and plots the MSE for validation and training across epochs.

    This function generates a line plot of the MSE values for validation and training from a CSV
    file named 'mse_hist.csv', located in a directory specified by the configuration dictionary.
    The plot is saved as an image file 'mse_hist.png' in the same directory.

    Args:
    config (dict): Configuration dictionary containing a nested dictionary 'simulation' which includes:
                   - 'savedir': A string specifying the directory path where the CSV file is located and
                                where the plot image will be saved.

    Returns:
    None

    Raises:
    FileNotFoundError: If the 'mse_hist.csv' file does not exist in the specified directory.
    Exception: For issues related to reading the CSV or plotting.
    """
    # Extract the directory path from the configuration dictionary
    savedir = config["simulation"]["savedir"]

    # Construct the full file path for the MSE history CSV file
    csv_path = os.path.join(savedir, "mse_hist.csv")

    try:
        # Load MSE data from the CSV file
        data = pd.read_csv(csv_path)

        # Create a line plot for MSE values
        plt.figure(figsize=(10, 6))  # Optional: Adjust figure size for better visibility
        plt.plot(data["epoch"], data["mse_val"], label="Validation")
        plt.plot(data["epoch"], data["mse_trn"], label="Training")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MSE per Epoch for Validation and Training")  # Optional: Add a title to the plot
        plt.legend()
        plt.grid(True)  # Optional: Add a grid for easier reading

        # Save the plot as an image file
        plt.savefig(os.path.join(savedir, "mse_hist.png"))

        # Close the plot to free up memory
        plt.close()

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {csv_path} does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while plotting MSE: {e}")

def visualize_histograms(config: dict):
    """
    Visualizes histograms for each layer of a neural network across different epochs,
    with interactive slider control to switch between epochs.

    Parameters:
    - config (dict): Configuration dictionary containing simulation parameters and weight distribution data.

    Returns:
    - None
    """
    # Extract simulation parameters and weight distribution data from config
    epochs = config["simulation"]["epochs"]
    layers = len(config["simulation"]["geometry"])
    bin_size = config["simulation"]["bin_size"]
    histogram_data = config["simulation"]["weight_distribution"]

    # Compute bin edges for histogram
    bins = np.arange(0, 1 + bin_size, bin_size)
    bin_edges = bins[:-1]

    # Create a figure with subplots, one for each layer
    fig, axs = plt.subplots(1, layers, squeeze=False)  # Use squeeze=False to ensure axs is always a 2D array

    # Plotting histograms for each layer
    bars = []
    for ax, layer_data in zip(axs.flat, range(layers)):
        counts = histogram_data[0][layer_data]  # Initial histogram data for the first epoch
        bar = ax.bar(bin_edges, counts, width=np.diff(bins), align='edge', edgecolor='black')
        bars.append(bar)
        ax.set_title(f'Layer {layer_data + 1}')
        ax.set_ylabel('Number of Weights' if layer_data == 0 else '')
        ax.set_xticks(bins)
        ax.set_xticklabels([f"{b:.2f}" for b in bins], rotation=45)

    # Create a slider for selecting epochs
    ax_slider = plt.axes([0.1, 0.05, 0.8, 0.03])  # Adjust position to not overlap with subplots
    slider = Slider(ax_slider, 'Epoch', 0, epochs - 1, valinit=0, valstep=1)

    # Define an update function for the slider
    def update(val):
        epoch_index = int(slider.val)
        for bar, layer_data in zip(bars, range(layers)):
            new_data = histogram_data[epoch_index][layer_data]
            for rect, count in zip(bar, new_data):
                rect.set_height(count)
            # Dynamically adjust the y-axis to accommodate the new data
            axs.flat[layer_data].set_ylim(0, max(new_data) * 1.1)

        fig.canvas.draw_idle()

    slider.on_changed(update)  # Connect the slider to the update function
    plt.show()