#!/usr/bin/env python3
"""Monitor running pilot experiments and send Telegram progress updates.

Checks the 3 agent output files every 5 minutes, parses progress,
and sends a consolidated Telegram message.
"""

import json
import re
import time
import urllib.request
from pathlib import Path

BOT_TOKEN = "7740709485:AAF35LkeavJ5-F4C6hcG5PC_7RdC9AeI8lI"
CHAT_ID = 7542082932
INTERVAL = 300  # 5 minutes

AGENTS = {
    "CALLM": "/private/tmp/claude-501/-Users-zwang/tasks/ab3786a6487673247.output",
    "V1": "/private/tmp/claude-501/-Users-zwang/tasks/affa481f4f6becdc4.output",
    "V2": "/private/tmp/claude-501/-Users-zwang/tasks/a5432e5b8dbcfdaba.output",
}

USERS = [71, 164, 119, 458, 310]
USER_ENTRIES = {71: 93, 164: 87, 119: 84, 458: 82, 310: 81}
TOTAL = 427


def send_telegram(text: str):
    try:
        data = json.dumps({"chat_id": CHAT_ID, "text": text}).encode()
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            data=data, headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"Telegram send failed: {e}")


def parse_progress(output_path: str) -> dict:
    """Parse agent output file to extract progress info."""
    result = {"current_user": None, "current_entry": 0, "total_entry": 0,
              "users_done": 0, "llm_calls": 0, "finished": False, "error": None}

    path = Path(output_path)
    if not path.exists():
        result["error"] = "output file not found"
        return result

    content = path.read_text(errors="ignore")

    # Check for completion
    if "PILOT COMPLETE" in content or "PILOT STUDY SUMMARY" in content:
        result["finished"] = True
        # Extract final LLM call count
        m = re.findall(r"LLM calls: (\d+)", content)
        if m:
            result["llm_calls"] = int(m[-1])
        # Extract metrics
        mae_match = re.search(r"Mean MAE: ([\d.]+)", content)
        if mae_match:
            result["mae"] = float(mae_match.group(1))
        ba_match = re.search(r"Mean Balanced Accuracy: ([\d.]+)", content)
        if ba_match:
            result["ba"] = float(ba_match.group(1))
        return result

    # Check for errors
    if "Error" in content and "Traceback" in content:
        lines = content.split("\n")
        for line in reversed(lines):
            if "Error" in line:
                result["error"] = line.strip()[:100]
                break

    # Find latest user
    user_matches = re.findall(r"--- User (\d+)", content)
    if user_matches:
        result["current_user"] = int(user_matches[-1])
        result["users_done"] = len(set(user_matches)) - 1  # current one is in progress

    # Find latest entry
    entry_matches = re.findall(r"Entry (\d+)/(\d+)", content)
    if entry_matches:
        result["current_entry"] = int(entry_matches[-1][0])
        result["total_entry"] = int(entry_matches[-1][1])

    # Count LLM calls
    call_matches = re.findall(r"Total LLM calls: (\d+)", content)
    if call_matches:
        result["llm_calls"] = int(call_matches[-1])

    return result


def format_progress(name: str, prog: dict) -> str:
    if prog["finished"]:
        extra = ""
        if "mae" in prog:
            extra += f" MAE={prog['mae']:.2f}"
        if "ba" in prog:
            extra += f" BA={prog['ba']:.2f}"
        return f"  {name}: DONE ({prog['llm_calls']} calls){extra}"

    if prog["error"]:
        return f"  {name}: ERROR - {prog['error']}"

    user = prog["current_user"] or "?"
    entry = prog["current_entry"]
    total = prog["total_entry"]

    # Calculate overall progress
    done_entries = 0
    if prog["current_user"]:
        for u in USERS:
            if u == prog["current_user"]:
                done_entries += entry
                break
            else:
                # Check if this user is done
                if prog["users_done"] > USERS.index(u) if u in USERS else False:
                    done_entries += USER_ENTRIES.get(u, 0)
                else:
                    done_entries += USER_ENTRIES.get(u, 0)

    pct = (done_entries / TOTAL * 100) if TOTAL > 0 else 0
    return f"  {name}: User {user} entry {entry}/{total} (~{pct:.0f}%)"


def main():
    send_telegram(
        "[proactive-affective-agent] Experiment monitor started\n"
        "3 parallel agents: CALLM, V1, V2\n"
        "5 users x 427 entries each\n"
        "Updates every 5 min"
    )

    all_done_sent = False

    while True:
        progresses = {}
        for name, path in AGENTS.items():
            progresses[name] = parse_progress(path)

        # Build message
        lines = ["[Pilot Experiment Progress]"]
        all_finished = True
        for name in ["CALLM", "V1", "V2"]:
            lines.append(format_progress(name, progresses[name]))
            if not progresses[name]["finished"]:
                all_finished = False

        msg = "\n".join(lines)
        print(f"[{time.strftime('%H:%M:%S')}] {msg}")
        send_telegram(msg)

        if all_finished and not all_done_sent:
            send_telegram(
                "[proactive-affective-agent] ALL 3 VERSIONS COMPLETE\n"
                "Check outputs/pilot/ for results.\n"
                "Open Claude Code for comparison analysis."
            )
            all_done_sent = True
            break

        time.sleep(INTERVAL)

    print("Monitor exiting - all experiments done.")


if __name__ == "__main__":
    main()
