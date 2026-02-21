"""Domain constants and mappings for the BUCS cancer survivorship study."""

# EMA trigger windows
EMA_WINDOWS = {
    "morning": ("08:00", "10:00"),
    "afternoon": ("13:00", "15:00"),
    "evening": ("19:00", "21:00"),
}

# Emotional state fields in EMA data
EMOTIONAL_STATE_FIELDS = [
    "valence",
    "arousal",
    "stress",
    "loneliness",
]

# Receptivity component fields
RECEPTIVITY_FIELDS = {
    "desire": "ER_desire_State",
    "availability": "INT_availability",
}

# Sensor file names
SENSOR_FILES = {
    "accelerometer": "ACCEL_2024-07-15.csv",
    "sleep": "SLEEP_2024-07-15.csv",
    "android_sleep": "AndroidSleep_2024-07-15.csv",
    "gps": "GPS_2024-07-15.csv",
    "app_usage": "APPUSAGE_2024-07-15.csv",
    "screen": "ScreenOnTime_2024-07-15 (1).csv",
    "motion": "MOTION_2024-07-15.csv",
    "key_input": "KeyInput_2024-07-15.csv",
}

# Number of cross-validation folds
NUM_FOLDS = 5
