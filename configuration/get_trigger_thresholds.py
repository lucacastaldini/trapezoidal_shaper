import os
import sys
sys.path.append('/home/gamma/workspace/gammalib/gammasim')
sys.path.append('/home/gamma/workspace/trapezoidal_shaper')
import numpy as np
import matplotlib.pyplot as plt
from gammasim import GammaSim
import json
from model.config_model import load_config, init_config
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

print(args.config_path)
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

absolute_path = os.path.abspath(args.config_path)
file_name      = os.path.basename(absolute_path)
dir_name       = os.path.dirname(absolute_path)
# configtps_path = os.path.join(dir_name, f'config_tps_{file_name}')

config_script_dir = os.path.dirname(os.path.abspath(__file__))
config_tps_file_path = os.path.join(config_script_dir, f'config_tps_{file_name}')
print("CONFIGGGGG:", config_tps_file_path)
cfg = None
if os.path.exists(config_tps_file_path):
    cfg = load_config(config_tps_file_path)
    print("Updating existent config: ", config_tps_file_path)
else:
    with open(args.config_path, 'r') as configfile:
        cfgsim = json.load(configfile)
        # initialize config trapz object
        cfg = init_config(args.config_path,
                          cfgsim['bkgbase_level'])
    print("Creting new config from default: ", config_tps_file_path)

################ Estimate trigger thresholds
## write to csv filter params and detection thresholds
cfg.time_filter.th_dy, cfg.time_filter.th_d2y = find_time_filter_params(
    gammasim, 
    tt, 
    cfg.time_filter.alpha_l, 
    cfg.time_filter.alpha_h, 
    cfg.time_filter.gain_k, 
    th_dy_multiplier=args.th_dy, th_d2y_multiplier=args.th_d2y)

# Ottieni il percorso della directory dello script

print(f'============> mori {config_tps_file_path} ')

# Salva la configurazione in un file JSON nella directory dello script
with open(config_tps_file_path, "w") as f:
    json.dump(cfg.model_dump(), f, indent=4)

print(f"File di configurazione salvato in: {config_tps_file_path}")

