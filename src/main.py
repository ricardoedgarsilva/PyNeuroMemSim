import time
import numpy as np

from tqdm import tqdm
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

    create_csv(config)

    print_information(config)
    copy_config_to_log(config)

    ltime = []
    for epoch in range(config["simulation"]["epochs"]):
        start_time = time.time()

        netlist.mk_circuit(config)
        netlist.save_net(config)

        run_ltspice(config, mode="-b")

        data = import_results(config)

        val_data, trn_data = split_data(data, len(x_test))

        mse_trn = calculate_mse(y_train, trn_data[-1])
        mse_val = calculate_mse(y_test, val_data[-1])

        updated_weights = backpropagate(
            config,
            trn_data,
            y_train,
            config["simulation"]["weights"],
            config["simulation"]["learning_rate"]
        )

        config["simulation"]["weights"] = updated_weights

        ltime.append(np.round(time.time() - start_time, 2))
        etime = np.round(np.mean(ltime) * (config["simulation"]["epochs"] - epoch), 2)	
        save_mse_hist(config, epoch, mse_trn, mse_val)
        print(f"Epoch {epoch}, MSE val: {mse_val}, MSE trn: {mse_trn}, Time: {ltime[-1]} s, ETA: {etime} s")

        
    print("Simulation finished!")
    plot_mse(config)
