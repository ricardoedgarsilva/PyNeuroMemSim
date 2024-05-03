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

def plot_weight_evolution(config: dict, weights: list):

    path = os.path.join(config["simulation"]["savedir"], "weight_evolution")
    os.makedirs(path)

    for layer_idx, layer in enumerate(weights):
        fig, ax = plt.subplots()
        cax = ax.matshow(layer, interpolation='nearest', cmap='bwr', vmin=0, vmax=1)
        fig.colorbar(cax)

        def update(epoch):
            cax.set_data(layer[epoch])
            ax.set_title(f'Epoch {epoch}')
            ax.set_aspect('equal', 'box')
            return cax,

        ani = FuncAnimation(fig, update, frames=config['simulation']['epochs'], repeat=True)
        ani.save(os.path.join(path, f"l{layer_idx}_weights.mp4"), writer='ffmpeg')

        plt.close(fig)
