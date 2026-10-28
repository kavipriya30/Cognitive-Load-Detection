"""
Feature Engineering Module for Cognitive Load Detection.

This module processes raw keystroke timing logs and extracts 15 core behavioral metrics:
1. typing_speed_wpm
2. characters_per_minute
3. average_key_hold_time
4. average_inter_key_interval
5. median_inter_key_interval
6. typing_time
7. pause_count
8. average_pause_duration
9. maximum_pause_duration
10. backspace_count
11. correction_rate
12. error_rate
13. typing_rhythm
14. keystrokes_per_second
15. total_keystrokes
"""

import numpy as np
from typing import Dict, List, Any


def levenshtein_distance(str1: str, str2: str) -> int:
    """Calculate the Levenshtein edit distance between two strings."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],       # Deletion
                                   dp[i][j - 1],       # Insertion
                                   dp[i - 1][j - 1])   # Substitution
    return dp[m][n]


def compute_typing_features(
    keydown_events: List[Dict[str, Any]],
    keyup_events: List[Dict[str, Any]],
    typed_text: str,
    target_text: str = "",
    pause_threshold_ms: float = 1500.0
) -> Dict[str, float]:
    """
    Extract 15 typing features from keydown/keyup event streams.

    :param keydown_events: List of dicts with 'key' and 'timestamp' (in ms)
    :param keyup_events: List of dicts with 'key' and 'timestamp' (in ms)
    :param typed_text: String typed by user
    :param target_text: Target prompt string (optional for error calculation)
    :param pause_threshold_ms: Milliseconds threshold to consider an inter-key delay as a pause (default 1.5s)
    :return: Dictionary containing all 15 engineered numerical features
    """

    # 1. Total Keystrokes & Backspace Count
    total_keystrokes = len(keydown_events)
    backspace_count = sum(1 for e in keydown_events if e.get('key') in ['Backspace', '8'])

    # 2. Timing Bounds
    if keydown_events:
        start_time = keydown_events[0]['timestamp']
        end_time = max(
            keydown_events[-1]['timestamp'],
            keyup_events[-1]['timestamp'] if keyup_events else keydown_events[-1]['timestamp']
        )
        typing_time_sec = max(0.1, (end_time - start_time) / 1000.0)
    else:
        typing_time_sec = 0.1

    # 3. Typing Speed Metrics
    char_count = len(typed_text)
    words_count = char_count / 5.0  # Standard WPM formula (5 characters = 1 word)
    typing_speed_wpm = (words_count / (typing_time_sec / 60.0)) if typing_time_sec > 0 else 0.0
    characters_per_minute = (char_count / (typing_time_sec / 60.0)) if typing_time_sec > 0 else 0.0
    keystrokes_per_second = (total_keystrokes / typing_time_sec) if typing_time_sec > 0 else 0.0

    # 4. Key Hold Times (keydown to keyup duration per key)
    key_holds = []
    # Match keydowns with keyups by key identifier sequentially
    keyup_pool = keyup_events.copy()
    for down in keydown_events:
        key_name = down.get('key')
        t_down = down.get('timestamp')
        # Find matching keyup
        match_idx = -1
        for idx, up in enumerate(keyup_pool):
            if up.get('key') == key_name and up.get('timestamp') >= t_down:
                match_idx = idx
                break
        if match_idx != -1:
            hold_duration = keyup_pool.pop(match_idx)['timestamp'] - t_down
            if 10.0 <= hold_duration <= 2000.0:  # Filter out extreme outliers
                key_holds.append(hold_duration)

    avg_key_hold_time = float(np.mean(key_holds)) if key_holds else 120.0

    # 5. Inter-key Intervals (Flight time between consecutive key press start times)
    inter_key_intervals = []
    for i in range(1, len(keydown_events)):
        interval = keydown_events[i]['timestamp'] - keydown_events[i - 1]['timestamp']
        if 0.0 <= interval <= 10000.0:
            inter_key_intervals.append(interval)

    if inter_key_intervals:
        avg_inter_key_interval = float(np.mean(inter_key_intervals))
        median_inter_key_interval = float(np.median(inter_key_intervals))
        typing_rhythm = float(np.std(inter_key_intervals))
    else:
        avg_inter_key_interval = 200.0
        median_inter_key_interval = 180.0
        typing_rhythm = 80.0

    # 6. Pauses Analysis
    pauses = [interval for interval in inter_key_intervals if interval >= pause_threshold_ms]
    pause_count = len(pauses)
    avg_pause_duration = float(np.mean(pauses)) if pauses else 0.0
    max_pause_duration = float(np.max(pauses)) if pauses else 0.0

    # 7. Correction & Error Rates
    correction_rate = backspace_count / max(1, total_keystrokes)

    if target_text and target_text.strip():
        dist = levenshtein_distance(typed_text.strip(), target_text.strip())
        error_rate = min(1.0, dist / max(1, len(target_text.strip())))
    else:
        error_rate = 0.0

    return {
        "typing_speed_wpm": round(typing_speed_wpm, 2),
        "characters_per_minute": round(characters_per_minute, 2),
        "average_key_hold_time": round(avg_key_hold_time, 2),
        "average_inter_key_interval": round(avg_inter_key_interval, 2),
        "median_inter_key_interval": round(median_inter_key_interval, 2),
        "typing_time": round(typing_time_sec, 2),
        "pause_count": pause_count,
        "average_pause_duration": round(avg_pause_duration, 2),
        "maximum_pause_duration": round(max_pause_duration, 2),
        "backspace_count": backspace_count,
        "correction_rate": round(correction_rate, 4),
        "error_rate": round(error_rate, 4),
        "typing_rhythm": round(typing_rhythm, 2),
        "keystrokes_per_second": round(keystrokes_per_second, 2),
        "total_keystrokes": total_keystrokes
    }


FEATURE_NAMES = [
    "typing_speed_wpm",
    "characters_per_minute",
    "average_key_hold_time",
    "average_inter_key_interval",
    "median_inter_key_interval",
    "typing_time",
    "pause_count",
    "average_pause_duration",
    "maximum_pause_duration",
    "backspace_count",
    "correction_rate",
    "error_rate",
    "typing_rhythm",
    "keystrokes_per_second",
    "total_keystrokes"
]
