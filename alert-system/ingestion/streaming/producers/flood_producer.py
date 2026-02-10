import requests
import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Locations avec IDs cohérents
locations = [
    {"id": "LOC_001", "lat": 34.02, "lon": -6.83, "name": "Rabat"},
    {"id": "LOC_002", "lat": 33.57, "lon": -7.59, "name": "Casablanca"},
    {"id": "LOC_003", "lat": 31.63, "lon": -8.00, "name": "Marrakech"},
    {"id": "LOC_004", "lat": 35.7595, "lon": -5.8340, "name": "Tanger"},
    {"id": "LOC_005", "lat": 33.8893, "lon": -5.5535, "name": "Fès"},
    {"id": "LOC_006", "lat": 30.4278, "lon": -9.5981, "name": "Agadir"},
    {"id": "LOC_007", "lat": 34.6814, "lon": -1.9086, "name": "Oujda"},
    {"id": "LOC_008", "lat": 33.2316, "lon": -8.5007, "name": "El Jadida"},
    {"id": "LOC_009", "lat": 35.1681, "lon": -5.2684, "name": "Tétouan"},
    {"id": "LOC_010", "lat": 32.2949, "lon": -9.2372, "name": "Safi"},
    {"id": "LOC_011", "lat": 34.2572, "lon": -6.5965, "name": "Kénitra"},
    {"id": "LOC_012", "lat": 33.5883, "lon": -7.6114, "name": "Mohammedia"},
    {"id": "LOC_013", "lat": 35.2595, "lon": -3.9366, "name": "Nador"},
    {"id": "LOC_014", "lat": 33.9715, "lon": -6.8498, "name": "Salé"},
    {"id": "LOC_015", "lat": 31.5085, "lon": -9.7595, "name": "Essaouira"},
]

static_data = {
    loc["id"]: {
        "elevation_m": random.randint(10, 800),
        "population_density": random.randint(100, 5000),
        "historical_floods": random.randint(0, 15),
        "infrastructure": random.choice(["low", "medium", "high"]),
        "soil_type": random.choice(["clay", "sand", "silt"]),
        "land_cover": random.choice(["urban", "forest", "agriculture"])
    }
    for loc in locations
}

print("🚀 Producer démarré...")

while True:
    for loc in locations:
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={loc['lat']}&longitude={loc['lon']}&current_weather=true"
            response = requests.get(url).json()
            weather = response["current_weather"]

            message = {
                "location_id": loc["id"],
                "latitude": loc["lat"],   # Envoyé directement
                "longitude": loc["lon"],  # Envoyé directement
                "temperature_c": weather["temperature"],
                "rainfall_mm": random.uniform(0, 200), 
                "humidity_pct": random.uniform(30, 100),
                "river_discharge_m3s": random.uniform(50, 5000),
                "water_level_m": random.uniform(0.5, 15),
                **static_data[loc["id"]],
                "timestamp": datetime.utcnow().isoformat()
            }

            producer.send("flood_stream", message)
            print(f"Sent data for {loc['id']} ({loc['lat']}, {loc['lon']})")
        except Exception as e:
            print(f"Error: {e}")

    time.sleep(10)