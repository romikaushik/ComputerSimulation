import numpy as np 
import random 
 
def format_array_as_string(array): 
    formatted_string = "" 
    for row in array: 
        formatted_string += " ".join(map(str, row)) + "\n"  
    return formatted_string.strip()  

def simulate_radioactive_decay(): 
    decay_constant = float(input("Enter decay constant (suggested: 0.02775): ")) 
    array_size = int(input("Enter the size of the 2D array (e.g., 50 for a 50x50 array): ")) 
    time_step = float(input("Enter the time step (suggested: 0.01): ")) 
 
    total_nuclei = array_size ** 2 
    half_life_actual = np.log(2) / decay_constant  
 
    nuclei = np.zeros((array_size, array_size), dtype=int) 
 
    initial_undecayed = total_nuclei 
    undecayed_count = initial_undecayed 
    time_elapsed = 0 
 
    while undecayed_count > initial_undecayed / 2: 
        for i in range(array_size): 
            for j in range(array_size): 
                if nuclei[i, j] == 0:  
                    if random.random() < decay_constant * time_step: 
                        nuclei[i, j] = 1  
 
        undecayed_count = np.sum(nuclei == 0) 
        time_elapsed += time_step 
 
    print("\nSimulation Results:") 
    print("Initial array of nuclei:") 
    print(format_array_as_string(np.zeros((array_size, array_size), dtype=int)))  
    print("\nFinal array of nuclei:") 
    print(format_array_as_string(nuclei))  
    print(f"\nInitial number of undecayed nuclei: {initial_undecayed}") 
    print(f"Final number of undecayed nuclei: {undecayed_count}") 
    print(f"Simulated half-life: {time_elapsed:.2f} minutes") 
    print(f"Actual half-life: {half_life_actual:.2f} minutes") 
 
simulate_radioactive_decay() 
