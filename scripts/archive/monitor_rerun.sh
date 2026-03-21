#!/usr/bin/env bash
# Monitor V2/V4 re-run progress, send Telegram at milestones
cd /Users/zwang/projects/proactive-affective-agent

MONITOR_LOG="outputs/pilot/logs/monitor.log"
STATE_FILE="outputs/pilot/logs/monitor_state.txt"

# Initialize state tracking
touch "$STATE_FILE"

send_telegram() {
    local msg="$1"
    local bot_token="${TELEGRAM_BOT_TOKEN:-}"
    local chat_id="${TELEGRAM_CHAT_ID:-7542082932}"
    if [ -z "$bot_token" ]; then return; fi
    python3 -c "
import json, subprocess
data = json.dumps({'chat_id': $chat_id, 'text': '''$msg'''})
subprocess.run(['curl', '-s', '-X', 'POST',
    'https://api.telegram.org/bot${bot_token}/sendMessage',
    '-H', 'Content-Type: application/json', '-d', data],
    capture_output=True)
" 2>/dev/null
}

get_total_entries() {
    local ver=$1 uid=$2
    local logfile="outputs/pilot/logs/rerun_${ver}_user${uid}.log"
    grep -oE "Total EMA entries: [0-9]+" "$logfile" 2>/dev/null | head -1 | grep -oE '[0-9]+$' || echo "0"
}

get_record_count() {
    local ver=$1 uid=$2
    local f="outputs/pilot/${ver}_user${uid}_records.jsonl"
    if [ -f "$f" ]; then
        wc -l < "$f" | tr -d ' '
    else
        echo "0"
    fi
}

get_entry_progress() {
    local ver=$1 uid=$2
    local logfile="outputs/pilot/logs/rerun_${ver}_user${uid}.log"
    grep -oE "Entry [0-9]+/[0-9]+" "$logfile" 2>/dev/null | tail -1 | sed 's/Entry //' || echo "0/?"
}

is_user_done() {
    local ver=$1 uid=$2
    local total=$(get_total_entries "$ver" "$uid")
    local count=$(get_record_count "$ver" "$uid")
    # Consider done if record count >= total - 2 (allowing for small discrepancies)
    if [ "$total" -gt 0 ] && [ "$count" -ge "$((total - 2))" ]; then
        echo "yes"
    else
        echo "no"
    fi
}

get_fallback_rate() {
    local ver=$1 uid=$2
    python3 -c "
import json
from pathlib import Path
f = Path('outputs/pilot/${ver}_user${uid}_records.jsonl')
if not f.exists():
    print('N/A')
else:
    entries = [json.loads(l) for l in f.read_text().strip().split('\n') if l.strip()]
    total = len(entries)
    fallback = sum(1 for e in entries if e.get('confidence', 1) <= 0.1)
    print(f'{fallback}/{total} ({fallback*100//max(total,1)}%)')
" 2>/dev/null
}

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Check if master script is still running
    master_running=$(ps aux | grep 'rerun_v2v4' | grep -v grep | wc -l | tr -d ' ')

    # V2 Batch 1 check
    v2b1_done=0
    for uid in 43 71 211 258 275; do
        if [ "$(is_user_done v2 $uid)" = "yes" ]; then
            ((v2b1_done++))
        fi
    done

    # V2 Batch 2 check
    v2b2_done=0
    for uid in 362 363 399 505 513 86; do
        if [ "$(is_user_done v2 $uid)" = "yes" ]; then
            ((v2b2_done++))
        fi
    done

    # V4 Batch 1 check
    v4b1_done=0
    for uid in 43 71 211 258 275; do
        if [ "$(is_user_done v4 $uid)" = "yes" ]; then
            ((v4b1_done++))
        fi
    done

    # V4 Batch 2 check
    v4b2_done=0
    for uid in 362 363 399 505 513; do
        if [ "$(is_user_done v4 $uid)" = "yes" ]; then
            ((v4b2_done++))
        fi
    done

    # Build status line
    status_line=""
    for uid in 43 71 211 258 275; do
        count=$(get_record_count v2 $uid)
        status_line="${status_line} v2u${uid}:${count}"
    done

    echo "[$timestamp] V2B1:${v2b1_done}/5 V2B2:${v2b2_done}/6 V4B1:${v4b1_done}/5 V4B2:${v4b2_done}/5 master=${master_running}${status_line}" >> "$MONITOR_LOG"

    # Read sent flags
    sent_v2b1=$(grep -c "SENT_V2B1" "$STATE_FILE" 2>/dev/null || echo 0)
    sent_v2all=$(grep -c "SENT_V2ALL" "$STATE_FILE" 2>/dev/null || echo 0)
    sent_v4b1=$(grep -c "SENT_V4B1" "$STATE_FILE" 2>/dev/null || echo 0)
    sent_final=$(grep -c "SENT_FINAL" "$STATE_FILE" 2>/dev/null || echo 0)

    # Milestone: V2 Batch 1 complete
    if [ "$v2b1_done" -eq 5 ] && [ "$sent_v2b1" -eq 0 ]; then
        details=""
        for uid in 43 71 211 258 275; do
            count=$(get_record_count v2 $uid)
            fb=$(get_fallback_rate v2 $uid)
            details="${details}
  user${uid}: ${count} records, fallback=${fb}"
        done
        msg="[pilot-monitor] V2 Batch 1 complete
Users 43,71,211,258,275 all finished.${details}
V2 Batch 2 starting next."
        send_telegram "$msg"
        echo "SENT_V2B1" >> "$STATE_FILE"
    fi

    # Milestone: All V2 complete
    v2_all=$((v2b1_done + v2b2_done))
    if [ "$v2_all" -ge 11 ] && [ "$sent_v2all" -eq 0 ]; then
        details=""
        for uid in 43 71 211 258 275 362 363 399 505 513 86; do
            count=$(get_record_count v2 $uid)
            fb=$(get_fallback_rate v2 $uid)
            details="${details}
  user${uid}: ${count} records, fallback=${fb}"
        done
        msg="[pilot-monitor] All V2 complete (11/11 users)${details}
V4 batches starting next."
        send_telegram "$msg"
        echo "SENT_V2ALL" >> "$STATE_FILE"
    fi

    # Milestone: V4 Batch 1 complete
    if [ "$v4b1_done" -eq 5 ] && [ "$sent_v4b1" -eq 0 ]; then
        details=""
        for uid in 43 71 211 258 275; do
            count=$(get_record_count v4 $uid)
            fb=$(get_fallback_rate v4 $uid)
            details="${details}
  user${uid}: ${count} records, fallback=${fb}"
        done
        msg="[pilot-monitor] V4 Batch 1 complete
Users 43,71,211,258,275 all finished.${details}
V4 Batch 2 starting next."
        send_telegram "$msg"
        echo "SENT_V4B1" >> "$STATE_FILE"
    fi

    # Milestone: ALL done
    all_done=$((v2b1_done + v2b2_done + v4b1_done + v4b2_done))
    if [ "$all_done" -ge 21 ] && [ "$sent_final" -eq 0 ]; then
        fallback_report=$(python3 -c "
import json
from pathlib import Path
users_v2 = [43, 71, 86, 211, 258, 275, 362, 363, 399, 505, 513]
users_v4 = [43, 71, 211, 258, 275, 362, 363, 399, 505, 513]
lines = []
for ver, users in [('v2', users_v2), ('v4', users_v4)]:
    for uid in users:
        f = Path(f'outputs/pilot/{ver}_user{uid}_records.jsonl')
        if not f.exists():
            lines.append(f'{ver} user{uid}: MISSING!')
            continue
        entries = [json.loads(l) for l in f.read_text().strip().split('\n') if l.strip()]
        total = len(entries)
        fallback = sum(1 for e in entries if e.get('confidence', 1) <= 0.1)
        pct = fallback*100//max(total,1)
        flag = ' WARN' if pct > 10 else ''
        lines.append(f'{ver} user{uid}: {total} entries, {fallback} fallback ({pct}%){flag}')
print('\n'.join(lines))
" 2>&1)
        msg="[pilot-monitor] ALL V2/V4 BATCHES COMPLETE
Total: 21/21 user-versions done.
Finished at: $(date)

${fallback_report}"
        send_telegram "$msg"
        echo "SENT_FINAL" >> "$STATE_FILE"
    fi

    # Check if script exited but not all done
    if [ "$master_running" -eq 0 ] && [ "$all_done" -lt 21 ] && [ "$sent_final" -eq 0 ]; then
        # Double check - wait one more cycle to be sure
        if grep -q "MASTER_EXIT_WARNED" "$STATE_FILE" 2>/dev/null; then
            msg="[pilot-monitor] WARNING: Master script exited prematurely!
Only $all_done/21 user-versions completed.
V2B1:${v2b1_done}/5 V2B2:${v2b2_done}/6 V4B1:${v4b1_done}/5 V4B2:${v4b2_done}/5
Check: tail outputs/pilot/logs/rerun_master.log"
            send_telegram "$msg"
            echo "SENT_FINAL" >> "$STATE_FILE"
        else
            echo "MASTER_EXIT_WARNED" >> "$STATE_FILE"
        fi
    fi

    # Exit if final sent
    if grep -q "SENT_FINAL" "$STATE_FILE" 2>/dev/null; then
        echo "[$timestamp] Monitoring complete. Exiting." >> "$MONITOR_LOG"
        exit 0
    fi

    sleep 180  # 3 minutes
done
