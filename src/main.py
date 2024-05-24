from modules.data_handling import *
from modules.dependencies import *
from modules.file_utils import *
from modules.netlist import *
from modules.plotting import *
from modules.simulation_utils import *
from config import *

if __name__ == "__main__":

    # Ensure that the save directory is in the src folder
    ensure_directory_is_src()

    # Create save directory
    config['simulation']['savedir'] = create_save_directory(
        config['simulation']['save']
    )
    
    # Create Netlist object
    netlist = Netlist()

    # Import data and create inputs
    x_train, y_train, x_test, y_test = import_data(config)
    create_inputs(x_train, x_test, config)

    # Calculate simulation time and initialize weights
    config["simulation"]["time"] = calculate_time(len(x_train), len(x_test), config)
    config["simulation"]["weights"] = initialize_weights(config)

    # Create MSE history CSV file and weight history HDF5 file
    create_mse_hist(config)

    # Print configuration and save config.log
    printlog_info(config)

    # Initialize list to store time
    ltime = []


    for epoch in range(config["simulation"]["epochs"]):
        start_time = time.time()

        # Print epoch cycle started
        print(f"\rEpoch {epoch} cycle started ....", end=' ')

        # Create netlist and save it
        netlist.mk_circuit(config)
        netlist.save_net(config)

        # Run LTSpice simulation
        run_ltspice(config, mode="-b")

        # Import results and split data
        data = import_results(config)

        # Split data into validation and training data
        val_data, trn_data = split_data(data, len(x_test))

        # Calculate MSE
        mse_trn = calculate_mse(y_train, trn_data[-1])
        mse_val = calculate_mse(y_test, val_data[-1])


        updated_weights = backpropagate(
            config,
            trn_data,
            y_train
        )

        config["simulation"]["weights"] = updated_weights


        ltime.append(np.round(time.time() - start_time, 2))
        etime = np.round(np.mean(ltime) * (config["simulation"]["epochs"] - epoch), 2)	
        append_mse_hist(config, epoch, mse_trn, mse_val)
        print(f"\rEpoch {epoch:4}, MSE val: {mse_val:0.5f}, MSE trn: {mse_trn:0.5f}, Time: {ltime[-1]:3.2f} s, Time Remaining: {etime:5.2f} s")

        
    print("Simulation finished!")
    plot_mse_hist(config)
