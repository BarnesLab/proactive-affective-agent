"""Domain constants and mappings for the BUCS cancer survivorship study."""

# EMA trigger windows
EMA_WINDOWS = {
    "morning": ("08:00", "10:00"),
    "afternoon": ("13:00", "15:00"),
    "evening": ("19:00", "21:00"),
}

# --- Prediction Targets ---

# Continuous targets (from EMA)
CONTINUOUS_TARGETS = {
    "PANAS_Pos": (0, 30),   # Positive affect aggregate
    "PANAS_Neg": (0, 30),   # Negative affect aggregate
    "ER_desire": (0, 10),   # Emotion regulation desire
}

# Binary state targets (individual-level)
BINARY_STATE_TARGETS = [
    "Individual_level_PA_State",
    "Individual_level_NA_State",
    "Individual_level_happy_State",
    "Individual_level_sad_State",
    "Individual_level_afraid_State",
    "Individual_level_miserable_State",
    "Individual_level_worried_State",
    "Individual_level_cheerful_State",
    "Individual_level_pleased_State",
    "Individual_level_grateful_State",
    "Individual_level_lonely_State",
    "Individual_level_interactions_quality_State",
    "Individual_level_pain_State",
    "Individual_level_forecasting_State",
    "Individual_level_ER_desire_State",
]

# Availability target
AVAILABILITY_TARGET = "INT_availability"

# All prediction target names
ALL_TARGETS = list(CONTINUOUS_TARGETS.keys()) + BINARY_STATE_TARGETS + [AVAILABILITY_TARGET]

# Emotional state fields in EMA data (raw scale items)
EMOTIONAL_STATE_FIELDS = [
    "happy", "cheerful", "pleased",
    "sad", "afraid", "miserable", "worried",
    "grateful", "lonely",
]

# Receptivity component fields
RECEPTIVITY_FIELDS = {
    "desire": "Individual_level_ER_desire_State",
    "availability": "INT_availability",
}

# --- Sensing Column Groups ---

SENSING_COLUMNS = {
    "accelerometer": {
        "file": "ACCEL_2024-07-15.csv",
        "date_col": "dt_feature",
        "id_col": "id_participant",
        "features": ["val_sleep_duration_min", "n_acc"],
        "description": "Accelerometer-based sleep detection",
    },
    "sleep": {
        "file": "SLEEP_2024-07-15.csv",
        "date_col": "dt_feature",
        "id_col": "id_participant",
        "features": ["amt_sleep_day_min"],
        "description": "Passive sleep tracking (duration in minutes)",
    },
    "android_sleep": {
        "file": "AndroidSleep_2024-07-15.csv",
        "date_col": "dt_feature",
        "id_col": "id_participant",
        "features": ["amt_sleep_min", "cat_status"],
        "description": "Android native sleep detection",
    },
    "gps": {
        "file": "GPS_2024-07-15.csv",
        "date_col": "dt_feature",
        "id_col": "id_participant",
        "features": [
            "n_capture_day", "n_travelevent_day",
            "amt_travel_day_km", "amt_travel_day_minutes",
            "amt_home_day_minutes", "amt_distancefromhome_day_max_km",
            "amt_location_day_variance", "amt_stoplocation_day_variance",
        ],
        "description": "GPS mobility and location patterns",
    },
    "app_usage": {
        "file": "APPUSAGE_2024-07-15.csv",
        "date_col": "dt_feature",
        "id_col": "id_participant",
        "features": ["id_app", "amt_foreground_day_sec"],
        "description": "App usage time per app",
    },
    "screen": {
        "file": "ScreenOnTime_2024-07-15 (1).csv",
        "date_col": "dt_feature",
        "id_col": "id_participant",
        "features": [
            "n_session_screenon_day", "amt_screenon_day_minutes",
            "amt_screenon_session_day_max_minutes",
            "amt_screenon_session_day_mean_minutes",
        ],
        "description": "Screen on/off patterns",
    },
    "motion": {
        "file": "MOTION_2024-07-15.csv",
        "date_col": "dt_feature",
        "id_col": "id_participant",
        "features": [
            "amt_keyboard_day_min", "amt_stationary_day_min",
            "amt_walking_day_min", "amt_automotive_day_min",
            "amt_running_day_min", "amt_cycling_day_min",
        ],
        "description": "Motion/activity classification (minutes per day)",
    },
    "key_input": {
        "file": "KeyInput_2024-07-15.csv",
        "date_col": "dt_feature",
        "id_col": "id_participant",
        "features": [
            "n_char_day_allapps", "n_word_day_allapps",
            "n_word_neg_day_allapps", "n_word_pos_day_allapps",
            "prop_word_neg_day_allapps", "prop_word_pos_day_allapps",
        ],
        "description": "Keyboard input sentiment analysis",
    },
}

# Legacy mapping (backward compat)
SENSOR_FILES = {name: info["file"] for name, info in SENSING_COLUMNS.items()}

# --- Baseline Trait Columns (key demographics & scales) ---

BASELINE_DEMOGRAPHICS = [
    "Study_ID", "age_demo", "gender", "race", "demo_ethnicity",
    "cancerdx", "cancer_years", "cancer_months", "cancer_stage",
    "education", "marital_status", "employment",
]

BASELINE_SCALES = [
    "PHQ8_TOTAL", "GAD7_TOTAL",
    "PANAS_POS", "PANAS_NEG",
    "TIPI_Extraversion", "TIPI_Agreeableness", "TIPI_Conscientiousness",
    "TIPI_Stability", "TIPI_Openness",
    "ERQ_Reappraisal", "ERQ_Suppression",
    "MSPSS_TOTAL", "GSE_TOTAL",
]

# Number of cross-validation folds
NUM_FOLDS = 5


# --- ID Conversion Helpers ---

def study_id_to_participant_id(study_id: int) -> str:
    """Convert Study_ID (int) to id_participant (zero-padded string)."""
    return str(study_id).zfill(3)


def participant_id_to_study_id(participant_id: str) -> int:
    """Convert id_participant (zero-padded string) to Study_ID (int)."""
    return int(participant_id)
