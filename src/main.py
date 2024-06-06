from modules.data_handling import *
from modules.dependencies import *
from modules.file_utils import *
from modules.learning_algorithms import *
from modules.metrics import *
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

    # Perform scaling and create inputs
    x_train, y_train, x_test, y_test = scale_data(config, x_train, y_train, x_test, y_test)

    create_inputs(x_train, x_test, config)

    # Calculate simulation time and initialize weights
    config["simulation"]["time"] = calculate_time(len(x_train), len(x_test), config)
    config["simulation"]["weights"] = initialize_weights(config)
    compute_weight_histogram(config)

    # Create metric history CSV file 
    create_mtr_hist(config)

    # Print configuration and save config.log
    printlog_info(config)

    # Initialize list to store time
    config["simulation"]["times"] = []


    for epoch in range(config["simulation"]["epochs"]):
        config["simulation"]["epoch"] = epoch
        stime = time.time()

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
        

        # Apply postprocessing if necessary
        metrics = calculate_metrics(config, trn_data[-1], val_data[-1], y_train, y_test)

        # Update weights based on the method
        func = globals().get(config["learning"]["algorithm"])
        if func:    func(config,trn_data,y_train)
        else:   raise ValueError(f"Algorithm {config['learning']['algorithm']} not found!")

        # Bound weights if necessary
        if config["learning"]["bound_weights"]: bound_weights(config)

        # Compute weight histogram
        compute_weight_histogram(config)
        save_weight_histogram(config)

        config["simulation"]["times"].append(np.round(time.time() - stime, 2))
        append_mtr_hist(config, metrics)
        print_metric_info(config, metrics)
        
    print("Simulation finished!")
    plot_mse_hist(config)
    visualize_histograms(config)
