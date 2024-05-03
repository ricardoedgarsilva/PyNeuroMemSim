from art import text2art

info = "\n".join([
    text2art("PyNeuroMemSim"),  # ASCII art title
    "Fully Connected Memristor based Neural Network Simulator\n",
    f" {10 * '-'} \n",
    "Author: Ricardo E. Silva",
    "Research Group: INESC MN, Lisbon, Portugal",
    "Licence: MIT",
    "Version: 0.3.4",
    "Channel: Experimental",
    f" {10 * '-'} \n\n\n",
    "Configuration:"
])


"""
Changelog:

• 0.1.0: Initial version
• 0.2.0: Added new experimental features
• 0.3.0: Added new logging capabilities
• 0.3.2: Added new non matricial backpropagation algorithm
• 0.3.3: 
    - Fix bug in info.py importation
    - Major restructuring of common.py
• 0.3.4:
    - Adding prints with '\r' to show feedback
    - Experimenting with different backpropagation
    
"""