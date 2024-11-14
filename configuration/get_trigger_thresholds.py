import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gammasim import GammaSim
import json
from configuration.model.config_model import load_config, cfg_instance
from implementation.filt_param_handlers import find_time_filter_params
import argparse

# Creazione dell'oggetto parser
parser = argparse.ArgumentParser(description="Fissa le Threshold in base alla configurazione di Gammasim")

# Argomento posizionale obbligatorio
parser.add_argument('config_path', type=str, help="Percorso del file di configurazione gammasim")

# Argomento opzionale con valore di default
parser.add_argument('th_dy', type=float, nargs='?', default=4, help="Valore di th_dy (default: 0.1)")

# Argomento opzionale con valore di default
parser.add_argument('th_d2y', type=float, nargs='?', default=0.2, help="Valore di th_d2y (default: 0.2)")

# Parsing degli argomenti
args = parser.parse_args()

# Visualizzazione dei valori ricevuti
print(f"Percorso configurazione: {args.config_path}")
print(f"Valore th_dy: {args.th_dy}")
print(f"Valore th_d2y: {args.th_d2y}")

config_file=args.config_path
saturation = False
gammasim = GammaSim(config_file)


gammasim.generate_dataset(saturation)

###prendo i dati - input 
vv = gammasim.get_dataset()[0]

dd = gammasim.get_sampling_time()
M = gammasim.get_params()[0][0]['tau2']
H1= gammasim.get_params()[0][0]['gamma']
t1 = gammasim.get_params()[0][0]['t_start']

### __init__
N = vv.shape[0]
nn = np.arange(0, N)
tt = nn * dd

cfg_name_extracted=args.config_path.split(".")[0]
file_name = f"config_tps_{cfg_name_extracted}.json"

if os.path.exists(file_name):
    cfg = load_config(file_name)
    print("Updating existent config: ", file_name)
else:
    cfg = cfg_instance
    print("Creting new config from default: ", file_name)

################ Estimate trigger thresholds
## write to csv filter params and detection thresholds
cfg.time_filter.th_dy, cfg.time_filter.th_d2y = find_time_filter_params(gammasim, tt, cfg.time_filter.alpha_l, cfg.time_filter.alpha_h, cfg.time_filter.gain_k, th_dy_multiplier=args.th_dy, th_d2y_multiplier=args.th_d2y)

# Salva la configurazione in un file JSON
with open("config_tps_"+cfg.gammasim_cfg.split('.')[0]+".json", "w") as f:
    json.dump(cfg.model_dump(), f, indent=4)

# Ottieni il percorso della directory dello script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, file_name)

# Salva la configurazione in un file JSON nella directory dello script
with open(file_path, "w") as f:
    json.dump(cfg.model_dump(), f, indent=4)

print(f"File di configurazione salvato in: {file_path}")

