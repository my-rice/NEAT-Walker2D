import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Funzione per leggere e combinare i dati
def read_and_combine_files(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    dfs = []
    file_names = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)
        file_name = os.path.splitext(os.path.basename(file))[0]  # Nome del file senza estensione
        # Estrai il numero dal nome del file se è presente
        file_number = ''.join(filter(str.isdigit, file_name))
        if file_number:
            file_names.append(f'Island_{file_number}')  # Aggiungi "Island_" al numero del file
        else:
            file_names.append('Island')  # Se non c'è un numero nel nome, usa solo "Island"
    # Ordina i dataframe e i nomi dei file in base al numero nel nome del file
    dfs_sorted, file_names_sorted = zip(*sorted(zip(dfs, file_names), key=lambda x: int(''.join(filter(str.isdigit, x[1])))))
    return dfs_sorted, file_names_sorted

# Funzione per creare un asse delle generazioni continuo
def extend_generations(df, max_gen):
    df['real_generation'] = df.apply(lambda row: row['generation'] + row['migration_step'] * max_gen, axis=1)
    return df

# Funzione per riempire i valori mancanti fino alla generazione massima
def fill_missing_generations(df, max_real_gen):
    last_fitness = df['fitness'].iloc[-1]
    max_real_gen = int(max_real_gen)  # Conversione a intero
    missing_gens = range(int(df['real_generation'].max()) + 1, max_real_gen + 1)
    missing_data = pd.DataFrame({'real_generation': missing_gens, 'fitness': last_fitness})
    df = pd.concat([df, missing_data])
    return df

# Funzione principale per processare una lista di cartelle specifiche
def process_specific_folders(folder_paths):
    max_fitness_dfs = []
    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        dfs, file_names = read_and_combine_files(folder_path)
        
        # Estensione delle generazioni
        max_gen = max(df['generation'].max() for df in dfs)
        dfs = [extend_generations(df, max_gen) for df in dfs]

        # Trova la generazione massima reale
        max_real_gen = max(df['real_generation'].max() for df in dfs)

        # Riempimento dei valori mancanti fino alla generazione massima
        dfs = [fill_missing_generations(df, max_real_gen) for df in dfs]
        
        # Identificazione del file con la fitness massima più alta per ciascuna cartella
        max_fitness_df = max(dfs, key=lambda df: df['fitness'].max())
        max_fitness_dfs.append(max_fitness_df)

    labels = ['Exp topology: RR', 'Exp topology: FF']  # Modifica con le etichette corrette   
    # Plotting della fitness per confronto
    plt.figure(figsize=(14, 7))
    i = 0
    for max_fitness_df, folder_path in zip(max_fitness_dfs, folder_paths):
        folder_name = os.path.basename(folder_path)
        max_fitness_value = max_fitness_df['fitness'].max()
        real_generation = max_fitness_df['real_generation'].values
        fitness = max_fitness_df['fitness'].values
        plt.plot(real_generation, fitness, label=labels[i], linewidth=2.5)
        i += 1
    
    # Evidenziazione delle migrazioni nei numeri dell'ascissa
    ax = plt.gca()
    migration_steps = sorted(set(step for df in max_fitness_dfs for step in df['migration_step'].unique()))
    migration_gen = [step * max_gen for step in migration_steps]

    # Disegna linee verticali corte e più in grassetto per i punti di migrazione
    for step in migration_steps:
        gen = step * max_gen
        if step == migration_steps[-1]:
            plt.axvline(x=gen, color='red', linestyle='--', linewidth=5, ymin=0, ymax=0.15)  # Linea corta rossa
        else:
            plt.axvline(x=gen, color='gray', linestyle='--', linewidth=5, ymin=0, ymax=0.15)  # Linea corta grigia

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness vs. Generations (Comparison of Max Values)')
    plt.legend()
    plt.grid(True)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    plt.savefig('comparison_FF.png')
    plt.close()

    # Calcolo della deviazione standard
    combined_df = pd.concat(max_fitness_dfs)
    std_df = combined_df.groupby('real_generation')['fitness'].std()

    # Plotting della deviazione standard
    plt.figure(figsize=(14, 7))
    plt.plot(std_df.index.values, std_df.values, label='Standard Deviation')
    plt.xlabel('Generations')
    plt.ylabel('Standard Deviation of Fitness')
    plt.title('Standard Deviation of Fitness vs. Generations')
    plt.legend()
    plt.grid(True)

    # Aggiungi le linee dei migration step
    for step in migration_steps:
        gen = step * max_gen
        if step == migration_steps[-1]:
            plt.axvline(x=gen, color='red', linestyle='--', linewidth=5, ymin=0, ymax=0.15)  # Linea corta rossa
        else:
            plt.axvline(x=gen, color='gray', linestyle='--', linewidth=5, ymin=0, ymax=0.15)  # Linea corta grigia

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    plt.savefig('comparison_std_plot.png')
    plt.close()

# Percorsi alle cartelle specifiche
folder_paths = [
    'Results_graph/STANDARD2-gen/',
    'Results_graph/STANDARD1-FF',
]  # Modifica con i percorsi corretti alle tue cartelle

# Esecuzione del processo per le cartelle specifiche
process_specific_folders(folder_paths)
