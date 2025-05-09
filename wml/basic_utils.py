import numpy as np
import colorama

def print_error(msg):
    print(colorama.Fore.RED+f"ERROR: {msg}"+colorama.Style.RESET_ALL)

def print_warning(msg):
    print(colorama.Fore.YELLOW+f"WARNING: {msg}"+colorama.Style.RESET_ALL)

def print_info(msg):
    print(colorama.Fore.BLUE+f"INFO: {msg}"+colorama.Style.RESET_ALL)

def np_unstack(array,axis=0):
    unstacked = np.split(array, array.shape[axis], axis=axis)
    unstacked = [arr.squeeze(axis=axis) for arr in unstacked]
    return unstacked