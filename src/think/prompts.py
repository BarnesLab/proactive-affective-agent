"""Prompt templates for CALLM + V1/V3 structured agent versions.

V2/V4 (agentic) build their prompts dynamically in cc_agent.py via
claude --print + MCP server — they don't use templates from here.

2x2 design:
                    Structured (fixed pipeline)    Agentic (autonomous tool-use)
  Sensing-only      V1 (prompts here)              V2 (cc_agent.py)
  Multimodal        V3 (prompts here)              V4 (cc_agent.py)

  CALLM: diary + TF-IDF RAG — CHI 2025 baseline (prompts here)
"""

from __future__ import annotations

from typing import Any

# Shared output format instructions
OUTPUT_FORMAT = """
You must respond with ONLY a JSON object (no markdown, no extra text) with these exact fields:

{
  "PANAS_Pos": <number 0-30, predicted positive affect>,
  "PANAS_Neg": <number 0-30, predicted negative affect>,
  "ER_desire": <number 0-10, predicted emotion regulation desire>,
  "Individual_level_PA_State": <boolean, is positive affect unusually high for this person?>,
  "Individual_level_NA_State": <boolean, is negative affect unusually high for this person?>,
  "Individual_level_happy_State": <boolean>,
  "Individual_level_sad_State": <boolean>,
  "Individual_level_afraid_State": <boolean>,
  "Individual_level_miserable_State": <boolean>,
  "Individual_level_worried_State": <boolean>,
  "Individual_level_cheerful_State": <boolean>,
  "Individual_level_pleased_State": <boolean>,
  "Individual_level_grateful_State": <boolean>,
  "Individual_level_lonely_State": <boolean>,
  "Individual_level_interactions_quality_State": <boolean>,
  "Individual_level_pain_State": <boolean>,
  "Individual_level_forecasting_State": <boolean>,
  "Individual_level_ER_desire_State": <boolean>,
  "INT_availability": <"yes" or "no", is user available for intervention?>,
  "reasoning": <string, brief explanation of your predictions>,
  "confidence": <number 0-1, your confidence in these predictions>
}

The "_State" fields indicate whether each measure is at an UNUSUAL level for this individual
compared to their typical baseline. True = unusual/elevated, False = typical.
""".strip()

CONTEXT_NOTE = """
Context: This is a cancer survivorship study. Participants are cancer survivors whose
emotional states and wellbeing are tracked via Ecological Momentary Assessment (EMA) surveys
multiple times daily. Your task is to predict their current emotional state and whether
they would be available for a just-in-time adaptive intervention.
""".strip()


# --- CALLM Baseline (CHI paper approach: diary text + RAG) ---

def callm_prompt(
    emotion_driver: str,
    rag_examples: str,
    memory_doc: str,
    trait_profile: str,
    date_str: str = "",
) -> str:
    """Build the CALLM baseline prompt (reactive, diary-text based).

    The CALLM approach from the CHI paper:
    1. Takes the user's diary text (emotion_driver)
    2. Retrieves similar cases from training data via TF-IDF
    3. Uses memory document for user context
    4. Makes predictions in a single LLM call
    """
    return f"""{CONTEXT_NOTE}

You are predicting the emotional state of a cancer survivor based on their diary entry.

## User Profile
{trait_profile}

## User Memory (longitudinal emotional trajectory)
{memory_doc[:3000] if memory_doc else "No memory document available."}

## Current Diary Entry{f' ({date_str})' if date_str else ''}
"{emotion_driver}"

## Similar Cases from Other Participants
These are diary entries from other participants with similar emotional expressions,
along with their actual emotional outcomes. Use these as reference:

{rag_examples}

## Task
Based on the diary entry, similar cases, and this user's history, predict their
current emotional state. Consider:
- What emotions does the diary text express?
- How do similar cases typically score?
- What is this user's typical baseline?

{OUTPUT_FORMAT}"""


# --- V1 Structured (sensing data + fixed pipeline) ---

def v1_system_prompt() -> str:
    """System prompt for V1 structured workflow."""
    return f"""{CONTEXT_NOTE}

You are an AI agent that predicts cancer survivors' emotional states from passive sensing data.
You follow a structured pipeline: analyze sensing data → consider user history → reason → predict.

{OUTPUT_FORMAT}"""


def v1_prompt(
    sensing_summary: str,
    memory_doc: str,
    trait_profile: str,
    date_str: str = "",
) -> str:
    """Build the V1 structured prompt (sensing-based, single LLM call).

    The V1 approach:
    1. Receives pre-formatted sensing data summary
    2. Considers memory document for longitudinal context
    3. Follows step-by-step reasoning to predict
    """
    return f"""## User Profile
{trait_profile}

## User Memory (longitudinal emotional trajectory)
{memory_doc[:3000] if memory_doc else "No memory document available."}

## Today's Passive Sensing Data{f' ({date_str})' if date_str else ''}
{sensing_summary}

## Instructions
Analyze the sensing data step by step:

1. **Sleep Analysis**: What do the sleep metrics suggest about rest quality?
2. **Mobility & Activity**: What do GPS, motion, and screen data reveal about activity level?
3. **Social Signals**: What do typing patterns and app usage suggest about social engagement?
4. **Pattern Integration**: How do these signals combine? Are there concerning patterns?
5. **User Context**: Given this user's history and traits, what would you predict?

Based on your analysis, provide your predictions as JSON.
{OUTPUT_FORMAT}"""


def format_sensing_summary(sensing_day) -> str:
    """Convert a SensingDay object to a natural language summary for prompts."""
    if sensing_day is None:
        return "No sensing data available for this day."

    data = sensing_day.to_summary_dict()
    if not data:
        return "No sensing data available for this day."

    sections = []

    # Sleep
    sleep_parts = []
    if data.get("accel_sleep_duration_min") is not None:
        sleep_parts.append(f"Accelerometer-detected sleep: {data['accel_sleep_duration_min']:.0f} min")
    if data.get("sleep_duration_min") is not None:
        sleep_parts.append(f"Passive sleep detection: {data['sleep_duration_min']:.0f} min")
    if data.get("android_sleep_min") is not None:
        status = f" ({data['android_sleep_status']})" if data.get("android_sleep_status") else ""
        sleep_parts.append(f"Android sleep: {data['android_sleep_min']:.0f} min{status}")
    if sleep_parts:
        sections.append("**Sleep:**\n" + "\n".join(f"  - {p}" for p in sleep_parts))

    # Mobility / GPS
    gps_parts = []
    if data.get("travel_km") is not None:
        gps_parts.append(f"Travel distance: {data['travel_km']:.1f} km")
    if data.get("travel_minutes") is not None:
        gps_parts.append(f"Travel time: {data['travel_minutes']:.0f} min")
    if data.get("home_minutes") is not None:
        gps_parts.append(f"Time at home: {data['home_minutes']:.0f} min")
    if data.get("max_distance_from_home_km") is not None:
        gps_parts.append(f"Max distance from home: {data['max_distance_from_home_km']:.1f} km")
    if data.get("location_variance") is not None:
        gps_parts.append(f"Location variance: {data['location_variance']:.4f}")
    if gps_parts:
        sections.append("**Mobility/GPS:**\n" + "\n".join(f"  - {p}" for p in gps_parts))

    # Activity / Motion
    activity_parts = []
    if data.get("stationary_min") is not None:
        activity_parts.append(f"Stationary: {data['stationary_min']:.0f} min")
    if data.get("walking_min") is not None:
        activity_parts.append(f"Walking: {data['walking_min']:.0f} min")
    if data.get("automotive_min") is not None:
        activity_parts.append(f"Driving: {data['automotive_min']:.0f} min")
    if data.get("running_min") is not None and data["running_min"] > 0:
        activity_parts.append(f"Running: {data['running_min']:.0f} min")
    if data.get("cycling_min") is not None and data["cycling_min"] > 0:
        activity_parts.append(f"Cycling: {data['cycling_min']:.0f} min")
    if activity_parts:
        sections.append("**Activity/Motion:**\n" + "\n".join(f"  - {p}" for p in activity_parts))

    # Screen
    screen_parts = []
    if data.get("screen_minutes") is not None:
        screen_parts.append(f"Total screen time: {data['screen_minutes']:.0f} min")
    if data.get("screen_sessions") is not None:
        screen_parts.append(f"Screen sessions: {data['screen_sessions']}")
    if data.get("screen_max_session_min") is not None:
        screen_parts.append(f"Longest session: {data['screen_max_session_min']:.0f} min")
    if screen_parts:
        sections.append("**Screen Usage:**\n" + "\n".join(f"  - {p}" for p in screen_parts))

    # Keyboard / Typing
    typing_parts = []
    if data.get("words_typed") is not None:
        typing_parts.append(f"Words typed: {data['words_typed']}")
    if data.get("prop_positive") is not None:
        typing_parts.append(f"Positive word ratio: {data['prop_positive']:.1%}")
    if data.get("prop_negative") is not None:
        typing_parts.append(f"Negative word ratio: {data['prop_negative']:.1%}")
    if typing_parts:
        sections.append("**Typing/Communication:**\n" + "\n".join(f"  - {p}" for p in typing_parts))

    # App usage
    if data.get("total_app_seconds") is not None:
        app_section = f"**App Usage:**\n  - Total foreground time: {data['total_app_seconds'] / 60:.0f} min"
        if data.get("top_apps"):
            top = ", ".join(f"{name} ({sec:.0f}s)" for name, sec in data["top_apps"][:3])
            app_section += f"\n  - Top apps: {top}"
        sections.append(app_section)

    return "\n\n".join(sections) if sections else "Minimal sensing data available."


# --- V3 Structured Full (diary + sensing + multimodal RAG) ---

def v3_system_prompt() -> str:
    """System prompt for V3 structured full workflow."""
    return f"""{CONTEXT_NOTE}

You are an AI agent that predicts cancer survivors' emotional states using BOTH
their diary entry AND passive sensing data, enhanced by similar cases from other
participants. You follow a structured 5-step pipeline.

{OUTPUT_FORMAT}"""


def v3_prompt(
    emotion_driver: str,
    sensing_summary: str,
    rag_examples: str,
    memory_doc: str,
    trait_profile: str,
    date_str: str = "",
) -> str:
    """Build the V3 structured full prompt (diary + sensing + multimodal RAG).

    V3 combines all data modalities with a fixed step-by-step reasoning pipeline:
    1. Diary analysis
    2. Sensing analysis
    3. Cross-modal consistency
    4. Similar case comparison (RAG with diary + sensing)
    5. Integrated prediction
    """
    return f"""## User Profile
{trait_profile}

## User Memory (longitudinal emotional trajectory)
{memory_doc[:3000] if memory_doc else "No memory document available."}

## Current Diary Entry{f' ({date_str})' if date_str else ''}
"{emotion_driver}"

## Today's Passive Sensing Data{f' ({date_str})' if date_str else ''}
{sensing_summary}

## Similar Cases from Other Participants (diary + sensing)
These are diary entries from other participants with similar emotional expressions,
along with their sensing data and actual emotional outcomes:

{rag_examples}

## Instructions
Analyze all available data following these 5 steps:

1. **Diary Analysis**: What emotions and concerns does the diary text express?
   Identify key emotional themes, coping language, and distress indicators.

2. **Sensing Analysis**: What do the passive sensing signals reveal?
   - Sleep patterns: rest quality and duration
   - Mobility: activity level, time away from home
   - Screen/typing: digital engagement patterns, sentiment in typing

3. **Cross-Modal Consistency**: Do diary content and sensing data tell a consistent
   story? Flag any discrepancies (e.g., diary reports feeling fine but sensing shows
   disrupted sleep and low mobility).

4. **Similar Case Comparison**: How do the retrieved cases with similar diary entries
   compare? Do their sensing patterns and outcomes align with this user's data?

5. **Integrated Prediction**: Synthesize all evidence — diary text, sensing signals,
   cross-modal patterns, similar cases, and user history — into a final prediction.

Based on your analysis, provide your predictions as JSON."""


# --- Shared helpers ---

def build_trait_summary(profile) -> str:
    """Build a concise trait summary string from a UserProfile."""
    if profile is None:
        return "No user profile available."
    return profile.to_text()
