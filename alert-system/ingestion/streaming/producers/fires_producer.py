import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer

# ===============================
# CONFIGURATION KAFKA
# ===============================
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

TOPIC = "environmental_stream"

# ===============================
# 15 villes au Maroc avec coordonnées
# ===============================
cities = [
    ("Rabat", 34.0209, -6.8416),
    ("Casablanca", 33.5731, -7.5898),
    ("Marrakech", 31.6295, -7.9811),
    ("Fes", 34.0333, -5.0000),
    ("Tanger", 35.7595, -5.8340),
    ("Agadir", 30.4278, -9.5981),
    ("Oujda", 34.6814, -1.9086),
    ("Tetouan", 35.5889, -5.3626),
    ("Safi", 32.2994, -9.2372),
    ("ElJadida", 33.2316, -8.5007),
    ("Kenitra", 34.2610, -6.5802),
    ("Nador", 35.1681, -2.9335),
    ("BeniMellal", 32.3373, -6.3498),
    ("Ouarzazate", 30.9189, -6.8934),
    ("Larache", 35.1932, -6.1557),
]

# ===============================
# Variables principales avec plages réalistes
# ===============================
variables_ranges = {
    "ELEV": (0, 2000),        # élévation en mètres
    "SLP": (0, 45),           # pente en degrés
    "EVT": (0, 100),          # évapotranspiration en mm
    "EVH": (0, 50),           # humidité d'évaporation
    "EVC": (0, 100),          # couverture végétale en %
    "CBD": (0, 1),           # densité de la canopée en kg/m3
    "CBH": (0, 30),           # hauteur de la canopée en m
    "CC": (0, 100),           # couverture nuageuse en %
    "CH": (0, 50),            # hauteur de la végétation en m
}

stats_types = ["max", "min", "median", "sum", "mode", "count", "mean"]

# ===============================
# Génération des statistiques pour une variable
# ===============================
def generate_variable_stats(var_name, min_val, max_val):
    """Génère toutes les statistiques pour une variable donnée"""
    stats = {}
    
    for stat in stats_types:
        if stat == "count":
            stats[f"{var_name}_{stat}"] = random.randint(50, 500)
        elif stat == "mode":
            # Pour mode, on génère une valeur aléatoire dans la plage
            stats[f"{var_name}_{stat}"] = round(random.uniform(min_val, max_val), 2)
        elif stat == "sum":
            # Pour sum, on génère une valeur plus grande
            stats[f"{var_name}_{stat}"] = round(random.uniform(min_val * 100, max_val * 100), 2)
        else:
            stats[f"{var_name}_{stat}"] = round(random.uniform(min_val, max_val), 2)
    
    return stats

# ===============================
# Génération d'un message complet
# ===============================
def generate_message(city, lat, lon):
    msg = {
        "Neighbour_acq_time": datetime.utcnow().isoformat(),
        "c_latitude": lat,
        "c_longitude": lon,
        "Shape": round(random.uniform(0.5, 1.5), 4),
    }
    
    # ===== Variables de base (self) =====
    for var, (min_val, max_val) in variables_ranges.items():
        msg.update(generate_variable_stats(var, min_val, max_val))
    
    # ===== Variables voisins =====
    neighbour_vars = list(variables_ranges.keys()) + ["Shape", "c_latitude"]
    for var in neighbour_vars:
        if var in variables_ranges:
            min_val, max_val = variables_ranges[var]
        elif var == "Shape":
            min_val, max_val = 0.5, 1.5
        elif var == "c_latitude":
            # Variation légère autour de la latitude de la ville
            min_val, max_val = lat - 0.5, lat + 0.5
        
        msg.update(generate_variable_stats(f"Neighbour_{var}", min_val, max_val))
    
    # ===== Variables météo =====
    msg.update({
        "TEMP_ave": round(random.uniform(10, 35), 2),
        "TEMP_min": round(random.uniform(0, 15), 2),
        "TEMP_max": round(random.uniform(25, 45), 2),
        "PRCP": round(random.uniform(0, 80), 2),
        "SNOW": round(random.uniform(0, 10), 2),
        "WDIR_ave": round(random.uniform(0, 360), 2),
        "WSPD_ave": round(random.uniform(0, 40), 2),
        "PRES_ave": round(random.uniform(980, 1035), 2),
        "WCOMP": round(random.uniform(-20, 20), 2),
    })
    
    # ===== Features dérivées - ratios =====
    msg["ELEV_ratio"] = msg["ELEV_mean"] / (msg["Neighbour_ELEV_mean"] + 1e-6)
    msg["SLP_ratio"] = msg["SLP_mean"] / (msg["Neighbour_SLP_mean"] + 1e-6)
    
    # ===== Features dérivées - ranges =====
    msg["ELEV_range"] = msg["ELEV_max"] - msg["ELEV_min"]
    msg["SLP_range"] = msg["SLP_max"] - msg["SLP_min"]
    
    # ===== Indices météo =====
    msg["fire_weather_index"] = msg["TEMP_max"] * msg["WSPD_ave"] / (msg["PRCP"] + 1)
    msg["dryness_index"] = msg["TEMP_ave"] / (msg["PRCP"] + 1)
    
    # ===== Interactions =====
    msg["veg_temp_interaction"] = msg["CH_median"] * msg["TEMP_max"]
    msg["lat_lon_interaction"] = lat * lon
    
    # ===== Diff, Ratio, Range pour toutes les variables =====
    for var in ["ELEV", "SLP", "EVT", "EVH", "EVC", "CBD", "CBH", "CC"]:
        # Différence
        msg[f"{var}_diff"] = msg[f"{var}_mean"] - msg[f"Neighbour_{var}_mean"]
        
        # Ratio
        msg[f"{var}_ratio"] = msg[f"{var}_mean"] / (msg[f"Neighbour_{var}_mean"] + 1e-6)
        
        # Range
        msg[f"{var}_range"] = msg[f"{var}_max"] - msg[f"{var}_min"]
    
    # Ajout du timestamp et location_id à la fin pour garder l'ordre
    msg.update({
        "location_id": city,
        "timestamp": datetime.utcnow().isoformat(),
        "processing_time": datetime.utcnow().isoformat(),
    })
    
    # Vérification que nous avons toutes les colonnes
    expected_columns = [
        "Neighbour_acq_time",
        "ELEV_max", "ELEV_min", "ELEV_median", "ELEV_sum", "ELEV_mode", "ELEV_count", "ELEV_mean",
        "SLP_max", "SLP_min", "SLP_median", "SLP_sum", "SLP_mode", "SLP_count", "SLP_mean",
        "EVT_max", "EVT_min", "EVT_median", "EVT_sum", "EVT_mode", "EVT_count", "EVT_mean",
        "EVH_max", "EVH_min", "EVH_median", "EVH_sum", "EVH_mode", "EVH_count", "EVH_mean",
        "EVC_max", "EVC_min", "EVC_median", "EVC_sum", "EVC_mode", "EVC_count", "EVC_mean",
        "CBD_max", "CBD_min", "CBD_median", "CBD_sum", "CBD_mode", "CBD_count", "CBD_mean",
        "CBH_max", "CBH_min", "CBH_median", "CBH_sum", "CBH_mode", "CBH_count", "CBH_mean",
        "CC_max", "CC_min", "CC_median", "CC_sum", "CC_mode", "CC_count", "CC_mean",
        "CH_max", "CH_min", "CH_median", "CH_sum", "CH_mode", "CH_count",
        "Shape", "c_latitude", "c_longitude",
        "Neighbour_ELEV_max", "Neighbour_ELEV_min", "Neighbour_ELEV_median", "Neighbour_ELEV_sum",
        "Neighbour_ELEV_mode", "Neighbour_ELEV_count", "Neighbour_ELEV_mean",
        "Neighbour_SLP_max", "Neighbour_SLP_min", "Neighbour_SLP_median", "Neighbour_SLP_sum",
        "Neighbour_SLP_mode", "Neighbour_SLP_count", "Neighbour_SLP_mean",
        "Neighbour_EVT_max", "Neighbour_EVT_min", "Neighbour_EVT_median", "Neighbour_EVT_sum",
        "Neighbour_EVT_mode", "Neighbour_EVT_count", "Neighbour_EVT_mean",
        "Neighbour_EVH_max", "Neighbour_EVH_min", "Neighbour_EVH_median", "Neighbour_EVH_sum",
        "Neighbour_EVH_mode", "Neighbour_EVH_count", "Neighbour_EVH_mean",
        "Neighbour_EVC_max", "Neighbour_EVC_min", "Neighbour_EVC_median", "Neighbour_EVC_sum",
        "Neighbour_EVC_mode", "Neighbour_EVC_count", "Neighbour_EVC_mean",
        "Neighbour_CBD_max", "Neighbour_CBD_min", "Neighbour_CBD_median", "Neighbour_CBD_sum",
        "Neighbour_CBD_mode", "Neighbour_CBD_count", "Neighbour_CBD_mean",
        "Neighbour_CBH_max", "Neighbour_CBH_min", "Neighbour_CBH_median", "Neighbour_CBH_sum",
        "Neighbour_CBH_mode", "Neighbour_CBH_count", "Neighbour_CBH_mean",
        "Neighbour_CC_max", "Neighbour_CC_min", "Neighbour_CC_median", "Neighbour_CC_sum",
        "Neighbour_CC_mode", "Neighbour_CC_count", "Neighbour_CC_mean",
        "Neighbour_CH_max", "Neighbour_CH_min", "Neighbour_CH_median", "Neighbour_CH_sum",
        "Neighbour_CH_mode", "Neighbour_CH_count", "Neighbour_Shape", "Neighbour_c_latitude",
        "TEMP_ave", "TEMP_min", "TEMP_max", "PRCP", "SNOW", "WDIR_ave", "WSPD_ave", "PRES_ave", "WCOMP",
        "ELEV_ratio", "SLP_ratio", "ELEV_range", "SLP_range", "fire_weather_index", "dryness_index",
        "veg_temp_interaction", "lat_lon_interaction",
        "ELEV_diff", "SLP_diff", "EVT_diff", "EVT_ratio", "EVT_range",
        "EVH_diff", "EVH_ratio", "EVH_range", "EVC_diff", "EVC_ratio", "EVC_range",
        "CBD_diff", "CBD_ratio", "CBD_range", "CBH_diff", "CBH_ratio", "CBH_range",
        "CC_diff", "CC_ratio", "CC_range",
        "location_id", "timestamp", "processing_time"
    ]
    
    # Vérification que nous avons toutes les colonnes
    missing = set(expected_columns) - set(msg.keys())
    if missing:
        print(f"⚠️ Colonnes manquantes pour {city}: {missing}")
        # Ajout des valeurs par défaut pour les colonnes manquantes
        for col in missing:
            if "CH_" in col and "count" not in col:
                msg[col] = 0  # CH_count est manquant dans la liste d'origine
    
    return msg

# ===============================
# Fonction pour afficher les colonnes générées
# ===============================
def print_message_structure(msg, city):
    print(f"\n📊 Structure du message pour {city}:")
    print(f"Nombre total de colonnes: {len(msg)}")
    
    # Grouper les colonnes par catégorie
    categories = {
        "Variables de base": [k for k in msg.keys() if any(v in k for v in ["ELEV", "SLP", "EVT", "EVH", "EVC", "CBD", "CBH", "CC", "CH"]) 
                              and "Neighbour_" not in k and "_diff" not in k and "_ratio" not in k and "_range" not in k],
        "Variables voisins": [k for k in msg.keys() if "Neighbour_" in k],
        "Variables météo": [k for k in msg.keys() if k in ["TEMP_ave", "TEMP_min", "TEMP_max", "PRCP", "SNOW", "WDIR_ave", "WSPD_ave", "PRES_ave", "WCOMP"]],
        "Features dérivées": [k for k in msg.keys() if "_diff" in k or "_ratio" in k or "_range" in k or 
                              "interaction" in k or "index" in k],
        "Informations location": [k for k in msg.keys() if k in ["location_id", "c_latitude", "c_longitude", "Shape", "timestamp", "processing_time", "Neighbour_acq_time"]]
    }
    
    for category, columns in categories.items():
        print(f"\n{category} ({len(columns)}):")
        print(f"  {', '.join(columns[:3])}..." if len(columns) > 3 else f"  {', '.join(columns)}")

# ===============================
# Loop principal
# ===============================
print("🚀 Fire Environmental Producer started")
print("=" * 80)

# Test avec une ville pour vérifier la structure
test_city = cities[0]
test_msg = generate_message(*test_city)
print_message_structure(test_msg, test_city[0])

print("\n" + "=" * 80)
print("📤 Envoi des données en continu...")
print("=" * 80)

counter = 0
while True:
    for city, lat, lon in cities:
        data = generate_message(city, lat, lon)
        producer.send(TOPIC, data)
        counter += 1
        
        if counter % 15 == 0:  # Afficher tous les 15 messages
            print(f"✅ Message {counter} envoyé pour {city}")
            print(f"   Timestamp: {data['timestamp']}")
            print(f"   Exemple de valeurs: TEMP_max={data['TEMP_max']:.2f}, PRCP={data['PRCP']:.2f}, fire_weather_index={data['fire_weather_index']:.2f}")
    
    time.sleep(10) 