import os

class Netlist:
    """Class for constructing and managing a SPICE netlist for circuit simulation."""

    def __init__(self):
        """Initialize an empty list to store the netlist."""
        self.circuit = []

    def add_end(self):
        """Add an end statement to the netlist."""
        self.circuit.append(".end")

    def add_transient(self, config):
        """Add transient analysis configuration to the netlist."""
        step = config['simulation']['timestep']
        time = config['simulation']['time'] + (20 * step)  
        self.circuit.append(f"\n.tran 0 {time} 0 {step} \n")
        self.circuit.append(".backanno\n")

    def add_libs(self, config):
        """Add library references and component models to the netlist."""
        self.circuit.append("\n.lib UniversalOpAmp1.lib\n")

        params = config["memristor"]["parameters"][config["memristor"]["model"]].copy()
        memristor_model = config["memristor"]["subcircuits"][config["memristor"]["model"]].format(**params)
        self.circuit.append(memristor_model)

    def add_save(self, config):
        """Add .save SPICE command for specified nodes in the netlist."""
        geometry = config['simulation']['geometry']

        savelst = [
            # Output nodes
            f"V(nin_{layer_i+1}_{col})" for layer_i, layer in enumerate(geometry) for col in range(layer[1])
        ] + [
            # Input nodes
            f"V(nin_{layer_i}_{row})" for layer_i, layer in enumerate(geometry) for row in range(layer[0])
            ]

        savelst_str = " ".join(savelst)
        self.circuit.append(f"\n.save {savelst_str}\n")

    def add_opamp_sources(self, config):
        """Add power supply sources for op-amps to the netlist."""
        power = config['opamp']['power']
        self.circuit.append("\n*** OpAmp Power Supplies ***\n")
        self.circuit.append(f"\nVdd nvdd 0 {power}")
        self.circuit.append(f"Vss nvss 0 {-power}")
        self.circuit.append(f"Vni nvni  0 {config['opamp']['noninverting']}")

    def add_vsource(self, id, nodep, nodem, value):
        """Add a voltage source to the netlist."""
        self.circuit.append(f"\nV{id} {nodep} {nodem} {value}\n")

    def add_vpwl(self, id, nodep, nodem, filedir):
        """Add a piecewise linear voltage source to the netlist."""
        self.circuit.append(f"\nVPWL{id} {nodep} {nodem} PWL file=\"{filedir}\"\n")

    def add_opamp(self, id, nodep, nodem, nodeo, vdd, vss):
        """Add an operational amplifier to the netlist."""
        self.circuit.append(f"\nX{id} {nodep} {nodem} {vdd} {vss} {nodeo} level1 Avol=1Meg GBW=10Meg Vos=0 En=0 Enk=0 In=0 Ink=0 Rin=500Meg\n")

    def add_resistor(self, id, nodep, nodem, value):
        """Add a resistor to the netlist."""
        self.circuit.append(f"\nR{id} {nodep} {nodem} {value}\n")

    def add_memristor(self, id, nodep, nodem, device, xo):
        """Add a memristor to the netlist."""
        self.circuit.append(f"\nX{id} {nodep} {nodem} XSV_{device} memristor xo={xo}")

    def mk_circuit(self, config):
        """
        Construct the complete netlist based on the provided configuration and save directory.
        
        Parameters:
        config (dict): Configuration dictionary for the circuit.
        """

        print("\rCreating circuit...", end=' ' * 20)

        # Clear the circuit
        self.circuit = []

        geometry = config['simulation']['geometry']
        savedir = config['simulation']['savedir']

        # Add the libraries
        self.add_libs(config)

        # Add the opamp power supplies
        self.add_opamp_sources(config)

        # Add the VPWL to the first layer
        self.circuit.append("\n*** Input Voltage ***\n")

        for row in range(geometry[0][0]):
            self.add_vpwl(
                f"IN_{row}", 
                f"nin_0_{row}", 
                0, 
                os.path.join(savedir, "inputs", f"in{row}.csv")
            )

        # Add the memristor to each layer
        for layer_i, layer in enumerate(geometry):
            self.circuit.append(f"\n*** Layer {layer_i} Memristors ***\n")
            for row in range(layer[0]):
                for col in range(layer[1]):
                    self.add_memristor(
                        f"M_{layer_i}_{row}_{col}", 
                        f"nin_{layer_i}_{row}", 
                        f"nout_{layer_i}_{col}", 
                        f"memristor_{layer_i}_{row}_{col}",
                        config["simulation"]["weights"][layer_i][row][col]
                    )   

        # Add the comparators to each layer
        for layer_i, layer in enumerate(geometry):
            self.circuit.append(f"\n*** Layer {layer_i} Comparators ***\n")
            for col in range(layer[1]):
                self.add_opamp(
                    f"UA_{layer_i}_{col}",
                    "nvni",
                    f"nout_{layer_i}_{col}",
                    f"noutb_{layer_i}_{col}",
                    "nvdd",
                    "nvss"
                )

                self.add_resistor(
                    f"A_{layer_i}_{row}_{col}",
                    f"nout_{layer_i}_{col}",
                    f"noutb_{layer_i}_{col}",
                    config["resistor"]["A"]
                )
                        
        # Add the unitary gain opamps to each layer
        for layer_i, layer in enumerate(geometry):
            self.circuit.append(f"\n*** Layer {layer_i} Unitary Gain Opamps ***\n")
            for col in range(layer[1]):

                self.add_resistor(
                    f"B_{layer_i}_{col}",
                    f"noutb_{layer_i}_{col}",
                    f"noutc_{layer_i}_{col}",
                    config["resistor"]["B"]
                )

                self.add_resistor(
                    f"C_{layer_i}_{col}",
                    f"noutc_{layer_i}_{col}",
                    f"nin_{layer_i+1}_{col}",
                    config["resistor"]["C"]
                )

                self.add_opamp(
                    f"UB_{layer_i}_{col}",
                    0,
                    f"noutc_{layer_i}_{col}",
                    f"nin_{layer_i+1}_{col}",
                    "nvdd",
                    "nvss"
                )

        # Transient analysis, save, and end statements
        self.add_transient(config)
        self.add_save(config)
        self.add_end()

        print("\rCircuit created!", end=' ' * 20)

    def save_net(self, config):
        """
        Save the constructed netlist to a file.

        Parameters:
        config (dict): Configuration dictionary for the circuit.
        """

        print("\rSaving netlist circuit file...", end=' ' * 20)
        savedir = config['simulation']['savedir']

        try:
            with open(os.path.join(savedir, "circuit.cir"), "w") as file:
                for line in self.circuit:
                    file.write(f"{line}\n")

            print("\rNetlist saved!", end=' ' * 20)

        except OSError as e:
            print(f"Error saving netlist: {e}")

