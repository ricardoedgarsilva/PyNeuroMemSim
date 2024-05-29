def create_weight_hist(config: dict, weights: list):
    path = os.path.join(config["simulation"]["savedir"], "weight_hist")
    os.makedirs(path)

    with h5py.File(os.path.join(path, "weights.hdf5"), "a") as f:
        for layer_idx, weight_array in enumerate(weights):

            # Create dataset 
            f.create_dataset(
                f'layer_{layer_idx}', 
                data = weight_array, 
                compression = "gzip",
                dtype = 'float32'
            )

            dataset = f[f'layer_{layer_idx}']
            dataset[0] = weight_array

def append_weight_hist(config: dict, weights: list):

    with h5py.File(os.path.join(config["simulation"]["savedir"], "weight_hist", "weights.hdf5"), "a") as f:
        for layer_idx, weight_array in enumerate(weights):
            dataset = f[f'layer_{layer_idx}']
            

            current_size = dataset.shape[0]
            new_size = current_size + 1
            dataset.resize(new_size, axis=0)

            dataset[current_size] = weight_array


