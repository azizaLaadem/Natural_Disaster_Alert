from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

client = InfluxDBClient(
    url="http://localhost:8086",
    token="BdeLJauoFfO2x4vP4jwtT9xBv5KDF4Nh2b5YvMPd7sa8izjhGhHzkF2JF3SfijlvAx7YsWxkAT-DFErvescglQ==",
    org="flood-monitoring"
)

write_api = client.write_api(write_options=SYNCHRONOUS)

point = Point("test_measurement").field("value", 1)

write_api.write(bucket="flood_predictions", record=point)
print("✅ Write OK")
