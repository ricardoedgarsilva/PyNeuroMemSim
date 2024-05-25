from art import text2art

info = "\n".join([
    text2art("PyNeuroMemSim"),  
    "Fully Connected Memristor based Neural Network Simulator\n",
    f" {10 * '-'} \n",
    "Author: Ricardo E. Silva",
    "Research Group: INESC MN, Lisbon, Portugal",
    "Licence: MIT",
    "Version: 0.3.6.5",
    "Channel: Experimental",
    f" {10 * '-'} \n\n\n",
    "Configuration:"
])


"""
Changelog:

• v0.1.0: Initial version
• v0.2.0: Added new experimental features
• v0.3.0: Added new logging capabilities
• v0.3.2: Added new non matricial backpropagation algorithm
• v0.3.3: 
    - Fix bug in info.py importation
    - Major restructuring of common.py
• v0.3.4:
    - Adding prints with '\r' to show feedback
    - Experimenting with different backpropagation
• v0.3.5:
    - Improving prints
    - Experimenting with different backpropagation functions
    - Added a new dataset
• v0.3.6:
    - Addew new dataset
    - Added new rounding conductivity value
    - More experimental features
• v0.3.6.5:
    - Remade the mnist dataset importation


       
"""