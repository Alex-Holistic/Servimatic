# Import libraries
import os
import csv
import json
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from datetime import *






# CLASS EDA INICIAL
def parse_json(mensaje):
    try:
        return json.loads(mensaje)
    except json.JSONDecodeError as e:
        print(f"Error en fila: {mensaje[:100]}... → {e}")
        return None


def get_main_keys(json_obj):
    if isinstance(json_obj, dict):
        return tuple(sorted(json_obj.keys()))
    return None

class EDAinicial:
    def __init__(self, filepath):
        """
        Classe per a fer un EDA inicial de fitxers de missatges. Aquesta classe segueix el següent procés:
        1. Carrega el fitxer CSV amb els missatges.
        2. Analitza la columna 'Mensaje' per extreure les claus principals dels objectes JSON.
        3. Filtra les files que contenen claus inesperades.
        4. Extreu les dades de 'PartAccount' en un DataFrame separat.
        5. Ordena les dades per 'FechaMensaje' i calcula les columnes 'Beneficis' i 'Credits'.
        6. Plotteja la evolució de 'Bet' i 'Win', així com 'Beneficis' i 'Credits' al llarg del temps i dels moviments.
        ---------------
        Args:
            filepath (str): Ruta al fitxer CSV amb els missatges d'una màquina concreta.

        """
        self.filepath = filepath
        self.df = None
        self.load_and_process_data()
    
    def load_and_process_data(self):
        
        self.df = pd.read_csv(self.filepath, sep=',')
        
        self.df['Missatge_json'] = self.df['Mensaje'].apply(parse_json)

        # only keep rows where 'MensajeTipoID' is 25
        self.df = self.df[self.df['MensajeTipoID'] == 25] # només missatges que contenen PartAccount
                      
        part_account_df = pd.json_normalize(self.df['Missatge_json'].apply(lambda x: x['PartAccount'] if isinstance(x, dict) and 'PartAccount' in x else {}))
        self.df = pd.concat([self.df[['ID', 'Maquina', 'FechaModificacion', 'FechaMensaje']].reset_index(drop=True), part_account_df.reset_index(drop=True)], axis=1)
        
        self.df['FechaMensaje'] = pd.to_datetime(self.df['FechaMensaje'], errors='coerce')
        self.df = self.df.sort_values(by='FechaMensaje')
        self.df.reset_index(inplace=True, drop =True)
        
        self.df['Beneficis']= self.df['Bet'] - self.df['Win']
        self.df['Credits']= self.df['CreditsIn'] - self.df['CreditsOut']


    def save_processed_data(self):
        """
        Desa el DataFrame processat en un fitxer CSV.
        
        """
        output_filepath = os.path.splitext(self.filepath)[0] + '_clean.csv'
        self.df.to_csv(output_filepath, index=False)
        print(f"Dades processades desades a: {output_filepath}")
        

    def plot_bet_win(self):
        """
        Genera un doble Plot en el que estudiarem la evolució de Bet i Win.
        Al primer plot veurem la evolució de Bet i Win al llarg del temps.
        Al segon plot veurem la evolució al llarg dels moviments.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19, 12))

        # Primer subplot: evolució al llarg del temps
        ax1.plot(self.df['FechaMensaje'], self.df['Bet'], label='Bet', color='blue')
        ax1.plot(self.df['FechaMensaje'], self.df['Win'], label='Win', color='orange')
        ax1.set_title('Evolució de Bet i Win al llarg del temps')
        ax1.set_xlabel('FechaMensaje')
        ax1.legend(title='Variables')
        ax1.grid()

        # Segon subplot: evolució al llarg dels moviments
        ax2.plot(self.df.index, self.df['Bet'], label='Bet', color='blue')
        ax2.plot(self.df.index, self.df['Win'], label='Win', color='orange')
        ax2.set_title(f'Evolució de Bet i Win al llarg dels {len(self.df)} moviments')
        ax2.set_xlabel('Número de moviment')
        ax2.legend(title='Variables')
        ax2.grid()

        plt.tight_layout()
        plt.show()
        

    def plot_beneficis_credits(self):
        """
        Genera un altre double plot en el que estudiarem la evolució de Beneficis i Credits.
        Al primer plot veurem la evolució de Beneficis i Credits al llarg del temps.
        Al segon plot veurem la evolució al llarg dels moviments. 
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(19, 12))

        # Primer subplot: evolució al llarg del temps
        ax1.plot(self.df['FechaMensaje'], self.df['Beneficis'], label='Beneficis (Bet - Win)', color='blue')
        ax1.plot(self.df['FechaMensaje'], self.df['Credits'], label='Credits (CreditsIn - CreditsOut)', color='orange', alpha=0.7)
        ax1.set_title(f'Evolució de Beneficis i Credits al llarg del temps')
        ax1.set_xlabel('FechaMensaje')
        ax1.legend(title='Variables')
        ax1.grid()
        

        # Segon subplot: evolució al llarg dels moviments
        ax2.plot(self.df.index, self.df['Beneficis'], label='Beneficis (Bet - Win)', color='blue')
        ax2.plot(self.df.index, self.df['Credits'], label='Credits (CreditsIn - CreditsOut)', color='orange', alpha=0.7)
        ax2.set_title(f'Evolució de Beneficis i Credits al llarg dels {len(self.df)} moviments')
        ax2.set_xlabel('Número de moviment')
        ax2.legend(title='Variables')
        ax2.grid()

        plt.tight_layout()
        plt.show()



# -----------------------------------------------------------------------------------------------------

# CLASSE PER ANALITZAR JSONS

class AnalitzadorEstructuresJSON:
    def __init__(self, carpeta_csv):
        self.carpeta_csv = carpeta_csv
        self.estructures_uniques = []  # Llista de mapes de claus derivades
        self.estructures_fitxers = defaultdict(list)  # Mapa: estructura_hashable -> llista de fitxers

    def analitzar_fitxers(self):
        """
        Analitza tots els fitxers CSV a la carpeta especificada i guarda estructures úniques.
        """
        for nom_fitxer in tqdm(os.listdir(self.carpeta_csv), desc="Analitzant fitxers CSV"):
            if nom_fitxer.endswith(".csv"):
                ruta_fitxer = os.path.join(self.carpeta_csv, nom_fitxer)
                self._analitzar_fitxer(ruta_fitxer)


    def _analitzar_fitxer(self, ruta_fitxer):
        """
        Analitza un fitxer CSV per extreure l'estructura JSON dels missatges de tipus 25.
        """
        try:
            df = pd.read_csv(ruta_fitxer)
            df_25 = df[df["MensajeTipoID"] == 25]

            if not df_25.empty:
                missatge_json = df_25.iloc[0]["Mensaje"]
                estructura = self._mapa_claus_derivades(missatge_json)
                estructura_hashable = self._estructura_com_hash(estructura)

                if estructura_hashable not in self.estructures_fitxers:
                    self.estructures_uniques.append(estructura)

                self.estructures_fitxers[estructura_hashable].append(os.path.basename(ruta_fitxer))
        except Exception as e:
            print(f"Error analitzant {ruta_fitxer}: {e}")

    def _mapa_claus_derivades(self, missatge):
        """
        Construeix un mapa de claus derivades a partir del JSON.
        """
        mapa = defaultdict(set)

        def recorre(dades, pare=None):
            if isinstance(dades, dict):
                for k, v in dades.items():
                    if pare:
                        mapa[pare].add(k)
                    recorre(v, k)
            elif isinstance(dades, list):
                for element in dades:
                    recorre(element, pare)

        try:
            dades = json.loads(missatge)
            recorre(dades)
            return {k: sorted(list(v)) for k, v in mapa.items()}
        except json.JSONDecodeError:
            return {"Error": ["JSON invàlid"]}

    def _estructura_com_hash(self, estructura):
        """
        Converteix una estructura en una representació hashable (tuple de tuples).
        """
        return tuple(sorted((k, tuple(sorted(v))) for k, v in estructura.items()))

    def mostrar_estructures_uniques(self):
        """
        Mostra totes les estructures úniques trobades.
        """
        print(f"\n🔎 S'han trobat {len(self.estructures_uniques)} estructures úniques:\n")
        for i, estructura in enumerate(self.estructures_uniques, 1):
            estructura_hashable = self._estructura_com_hash(estructura)
            fitxers = self.estructures_fitxers[estructura_hashable]
            print(f"🧩 Estructura {i} ({len(fitxers)} fitxers):")
            for clau, derivades in estructura.items():
                print(f"  {clau}: {', '.join(derivades)}")
            print()

    def mostrar_fitxers_per_estructura(self):
        """
        Mostra quins fitxers pertanyen a cada estructura única.
        """
        print(f"\n📂 Fitxers agrupats per estructura:\n")
        for i, estructura in enumerate(self.estructures_uniques, 1):
            estructura_hashable = self._estructura_com_hash(estructura)
            fitxers = self.estructures_fitxers[estructura_hashable]
            print(f"🧩 Estructura {i} ({len(fitxers)} fitxers):")
            print(f"  Fitxers: {', '.join(fitxers)}\n")

    def buscar_claus_derivades(self, clau_principal):
        """
        Mostra les claus derivades d'una clau principal en totes les estructures úniques.
        """
        print(f"\n🔍 Claus derivades per a la clau '{clau_principal}':\n")
        for i, estructura in enumerate(self.estructures_uniques, 1):
            if clau_principal in estructura:
                derivades = estructura[clau_principal]
                print(f"🧩 Estructura {i}: {', '.join(derivades)}")
            else:
                print(f"🧩 Estructura {i}: ❌ Clau no trobada")

    
    def mostrar_json_maquina(self, nom_fitxer):
        """
        Mostra l'estructura JSON (claus derivades) d'una màquina concreta. L'input és el nom del fitxer sense l'extensió .csv.
        
        """
        try:
            ruta_fitxer = os.path.join(self.carpeta_csv, nom_fitxer+'.csv')
            df = pd.read_csv(ruta_fitxer)
            df_25 = df[df["MensajeTipoID"] == 25]

            if not df_25.empty:
                missatge_json = json.loads(df_25.iloc[0]["Mensaje"])               
                print(f"\n📄 Estructura JSON per a la màquina '{nom_fitxer}':\n")
                pprint(missatge_json, sort_dicts=True)
            else:
                print(f"⚠️ No s'ha trobat cap missatge de tipus 25 al fitxer '{nom_fitxer}'.")
        except Exception as e:
            print(f"❌ Error llegint el fitxer '{nom_fitxer}': {e}\nRecorda que el nom del fitxer NO ha d'incloure l'extensió .csv.")
            print("Exemple de input correcte: '7PO2.2034.25'")

        

# -----------------------------------------------------------------------------------

# Funció per afegir data als fitxers amb Missatges per màquina


import os
import csv
from tqdm import tqdm


def clean_csv_raw_Mensajes(n):
    full_dataset_path = os.path.join(os.path.dirname(os.getcwd()), 'data','MensajesRAW', f'Mensajes_{n}.csv')
    columns = [
        "ID", "MsgID", "FechaMensaje", "ReqID", "SgID", "Maquina", "SGVer",
        "ConexionTipoID", "MensajeTipoID", "Mensaje", "FechaModificacion", "UsuarioModificacion"
    ]
    full_df= pd.read_csv(full_dataset_path, sep=';', names = columns, header=None)
    full_df.to_csv(os.path.join(os.path.dirname(os.getcwd()), 'data', f'master_mensajes_{n}.csv'), index=False)


def split_csv_by_machine_buffered(input_file_path, buffer_size=500):
    columns = [
    "ID", "MsgID", "FechaMensaje", "ReqID", "SgID", "Maquina", "SGVer",
    "ConexionTipoID", "MensajeTipoID", "Mensaje", "FechaModificacion", "UsuarioModificacion"
    ]
    output_dir = os.path.join(os.path.dirname(input_file_path), 'Missatges_x_maquina')
    os.makedirs(output_dir, exist_ok=True)

    buffers = {}  # {machine_id: [rows]}

    def flush_buffer(machine_id):
        """Write buffered rows to the corresponding file and clear buffer."""
        machine_file_path = os.path.join(output_dir, f'{machine_id}.csv')
        existing_ids = set()

        # Load existing IDs if file exists
        if os.path.exists(machine_file_path):
            with open(machine_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # skip header
                for r in reader:
                    if r:
                        existing_ids.add(r[0])  # assuming first column is ID

        # Filter out duplicates
        new_rows = [row for row in buffers[machine_id] if row[0] not in existing_ids]

        if not new_rows:
            buffers[machine_id] = []
            return

        write_header = not os.path.exists(machine_file_path)
        with open(machine_file_path, 'a', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, delimiter=',')
            if write_header:
                writer.writerow(columns)
            writer.writerows(new_rows)

        buffers[machine_id] = []

    # Count total lines for progress bar
    total_lines = sum(1 for _ in open(input_file_path, 'r', encoding='utf-8', errors='ignore')) - 1  # minus header

    with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.reader(infile, delimiter=',')
        next(reader, None)  # Skip header row

        for row in tqdm(reader, total=total_lines, desc="Processing rows"):
            if len(row) < 6:
                continue
            machine_id = row[5].strip()
            if not machine_id:
                continue

            if machine_id not in buffers:
                buffers[machine_id] = []

            buffers[machine_id].append(row)

            # Flush buffer if it reaches buffer_size
            if len(buffers[machine_id]) >= buffer_size:
                flush_buffer(machine_id)

    # Flush remaining buffers
    for machine_id in buffers:
        if buffers[machine_id]:
            flush_buffer(machine_id)

    print("✅ Split completed successfully!")