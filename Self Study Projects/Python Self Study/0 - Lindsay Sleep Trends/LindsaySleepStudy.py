import pandas as pd
import matplotlib.pyplot as plt
import os

# Load CSV
df = pd.read_csv("data/sleep_sessions.csv", parse_dates=["STARTTIME", "ENDTIME"])
df = df.sort_values("STARTTIME")  # ensure chronological order

# --- 1. Sleep Quality Over Time ---
plt.figure(figsize=(10, 5))
plt.plot(df["STARTTIME"], df["SLEEPQUALITY"], marker='o', color='steelblue')
plt.title("Sleep Quality Over Time")
plt.xlabel("Date")
plt.ylabel("Sleep Quality (0–1)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 2. Stacked Sleep Stage Durations ---
plt.figure(figsize=(10, 5))
plt.stackplot(df["STARTTIME"],
              df["TIMEINDEEPSLEEP"] / 3600,
              df["TIMEINLIGHTSLEEP"] / 3600,
              df["TIMEINREMSLEEP"] / 3600,
              labels=['Deep Sleep', 'Light Sleep', 'REM Sleep'],
              colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title("Sleep Stages Over Time")
plt.xlabel("Date")
plt.ylabel("Hours")
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 3. Time to Fall Asleep ---
plt.figure(figsize=(10, 5))
plt.plot(df["STARTTIME"], df["TIMETOSLEEP"] / 60, marker='x', linestyle='-', color='purple')
plt.title("Time to Fall Asleep")
plt.xlabel("Date")
plt.ylabel("Minutes")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 4. Number of Awakenings ---
plt.figure(figsize=(10, 5))
plt.bar(df["STARTTIME"], df["NUMBEROFAWAKENINGS"], color='salmon')
plt.title("Number of Awakenings")
plt.xlabel("Date")
plt.ylabel("Awakenings")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# --- 5. Sleep Quality vs Time Awake ---
plt.figure(figsize=(7, 7))
plt.scatter(df["TIMEAWAKE"] / 60, df["SLEEPQUALITY"], color='teal')
plt.title("Sleep Quality vs. Time Awake")
plt.xlabel("Time Awake (minutes)")
plt.ylabel("Sleep Quality (0–1)")
plt.grid(True)
plt.tight_layout()
plt.show()
