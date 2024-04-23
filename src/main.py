import numpy as np


from modules.netlist import Netlist
from modules.common import *
from config import *


if __name__ == "__main__":

    ensure_directory_is_src()

    config['simulation']['savedir'] = create_save_directory(
        config['simulation']['save']
    )
    
    # Create Netlist object
    netlist = Netlist()
    # Import data and create inputs
    x_train, y_train, x_test, y_test = import_data(config)
    create_inputs(x_train, x_test, config)

    config["simulation"]["time"] = calculate_time(len(x_train), len(x_test), config)
    config["simulation"]["weights"] = initialize_weights(config)

    print_information(config)

    for epoch in range(config["simulation"]["epochs"]):

        netlist.mk_circuit(config)
        netlist.save_net(config)

        run_ltspice(config, mode="-b")

        data = import_results(config)

        updated_weights, mse_val, mse_trn = update_weights(
            data,
            y_test,
            y_train, 
            config["simulation"]["weights"], 
            config["simulation"]["learning_rate"]
        )

        config["simulation"]["weights"] = updated_weights
        print(f"Epoch {epoch}, MSE val: {mse_val}, MSE trn: {mse_trn}")

        
    print("Simulation finished!")
