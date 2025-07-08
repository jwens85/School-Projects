import pandas as pd
import re
from datetime import datetime, timedelta

# Path to raw data file
data_path = "data/LindsayPillowData.txt"

# Read in the raw text
with open(data_path, "r", encoding="utf-8") as file:
    raw_text = file.read()

# Extract sleep session blocks
pattern = re.compile(r"Z_PK\s*->\s*\d+Z_ENT\s*->\s*6.*?ZUNIQUEIDENTIFIER\s*->\s*[A-Z0-9-]+", re.DOTALL)
sessions = pattern.findall(raw_text)

# Parse into dictionaries
records = []
for session in sessions:
    record = {}
    for match in re.findall(r"Z([A-Z0-9_]+)\s*->\s*(-?\d+(?:\.\d+)?)", session):
        key, val = match
        record[key] = float(val)
    records.append(record)

# Convert to DataFrame
df = pd.DataFrame(records)

# Convert Apple absolute times to datetime (origin: Jan 1, 2001)
def mac_time_to_datetime(val):
    try:
        return datetime(2001, 1, 1) + timedelta(seconds=val)
    except:
        return pd.NaT

for col in ["STARTTIME", "ENDTIME"]:
    if col in df.columns:
        df[col] = df[col].apply(mac_time_to_datetime)

# Preview
print(df[["STARTTIME", "ENDTIME", "SLEEPQUALITY", "TIMEINDEEPSLEEP", "TIMEINLIGHTSLEEP", "TIMEINREMSLEEP"]].head())

# Optional: save to CSV
df.to_csv("sleep_sessions.csv", index=False)
