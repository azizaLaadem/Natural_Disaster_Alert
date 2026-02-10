from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timezone

client = InfluxDBClient(
    url="http://localhost:8086",
    token="BdeLJauoFfO2x4vP4jwtT9xBv5KDF4Nh2b5YvMPd7sa8izjhGhHzkF2JF3SfijlvAx7YsWxkAT-DFErvescglQ==",
    org="flood-monitoring"
)

write_api = client.write_api(write_options=SYNCHRONOUS)

point = (
    Point("debug_test")
    .tag("test", "yes")
    .field("value", 1.0)
    .time(datetime.now(timezone.utc))
)

write_api.write(
    bucket="flood_predictions",
    record=point
)

print("WRITE DONE")
