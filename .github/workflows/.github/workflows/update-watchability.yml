import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.collegefootballdata.com"
OUTPUT_PATH = Path("data/watchability_index.csv")

# Your current formula:
RANKING_WEIGHT = 0.55
SPREAD_WEIGHT = 0.45

# A lower value means unranked games receive less ranking credit.
UNRANKED_TEAM_SCORE = 15.0


def api_get(endpoint: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Request data from CollegeFootballData and return its JSON list."""

    api_key = os.environ.get("CFBD_API_KEY")

    if not api_key:
        raise RuntimeError(
            "CFBD_API_KEY is missing. Add it as a GitHub Actions secret."
        )

    response = requests.get(
        f"{BASE_URL}{endpoint}",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=60,
    )

    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list):
        raise TypeError(
            f"Expected a list from {endpoint}, received {type(payload).__name__}."
        )

    return payload


def rank_to_score(rank: int | None) -> float:
    """
    Convert an AP ranking into a 0–100 quality score.

    #1 receives 98 points.
    #25 receives 26 points.
    Unranked teams receive 15 points.
    """

    if rank is None:
        return UNRANKED_TEAM_SCORE

    return max(0.0, 101.0 - (rank * 3.0))


def spread_to_score(spread: float | None) -> float | None:
    """
    Convert the absolute point spread into a competitiveness score.

    Pick'em = 100.
    7-point spread = 72.
    14-point spread = 44.
    25 points or more = 0.
    """

    if spread is None:
        return None

    return max(0.0, 100.0 - (abs(spread) * 4.0))


def extract_ap_rankings(
    ranking_data: list[dict[str, Any]],
) -> dict[str, int]:
    """Create a dictionary such as {'Georgia': 3, 'Ohio State': 5}."""

    rankings: dict[str, int] = {}

    for weekly_entry in ranking_data:
        for poll in weekly_entry.get("polls", []):
            poll_name = str(poll.get("poll", "")).lower()

            if "ap" not in poll_name:
                continue

            for ranked_team in poll.get("ranks", []):
                school = ranked_team.get("school")
                rank = ranked_team.get("rank")

                if school and rank is not None:
                    rankings[str(school)] = int(rank)

    return rankings


def select_spread(line_entry: dict[str, Any]) -> float | None:
    """
    Select one consistent spread.

    Preference order:
    1. Consensus
    2. DraftKings
    3. ESPN Bet
    4. First available provider
    """

    lines = line_entry.get("lines") or []

    if not lines:
        return None

    preferred_providers = [
        "consensus",
        "draftkings",
        "espn bet",
        "espnbet",
    ]

    for preferred in preferred_providers:
        for line in lines:
            provider = str(line.get("provider", "")).lower()

            if preferred in provider and line.get("spread") is not None:
                return float(line["spread"])

    for line in lines:
        if line.get("spread") is not None:
            return float(line["spread"])

    return None


def build_week(
    season: int,
    week: int,
    season_type: str = "regular",
) -> pd.DataFrame:
    """Pull and combine games, AP rankings and point spreads for one week."""

    common_params = {
        "year": season,
        "week": week,
        "seasonType": season_type,
    }

    games = api_get("/games", common_params)
    ranking_data = api_get("/rankings", common_params)
    betting_data = api_get("/lines", common_params)

    ap_rankings = extract_ap_rankings(ranking_data)

    # Match betting lines to the CollegeFootballData game ID.
    betting_by_game_id = {
        int(entry["id"]): entry
        for entry in betting_data
        if entry.get("id") is not None
    }

    rows: list[dict[str, Any]] = []

    for game in games:
        game_id = game.get("id")
        home_team = game.get("home_team")
        away_team = game.get("away_team")

        if game_id is None or not home_team or not away_team:
            continue

        home_rank = ap_rankings.get(str(home_team))
        away_rank = ap_rankings.get(str(away_team))

        home_rank_score = rank_to_score(home_rank)
        away_rank_score = rank_to_score(away_rank)

        ranking_quality = (home_rank_score + away_rank_score) / 2.0

        line_entry = betting_by_game_id.get(int(game_id), {})
        spread = select_spread(line_entry)
        spread_score = spread_to_score(spread)

        # Do not pretend a missing betting line equals a pick'em.
        watchability = (
            (ranking_quality * RANKING_WEIGHT)
            + (spread_score * SPREAD_WEIGHT)
            if spread_score is not None
            else None
        )

        rows.append(
            {
                "season": season,
                "season_type": season_type,
                "week": week,
                "game_id": int(game_id),
                "start_date": game.get("start_date"),
                "away_team": away_team,
                "home_team": home_team,
                "game": f"{away_team} at {home_team}",
                "away_conference": game.get("away_conference"),
                "home_conference": game.get("home_conference"),
                "away_rank": away_rank,
                "home_rank": home_rank,
                "away_rank_display": (
                    f"#{away_rank}" if away_rank is not None else "NR"
                ),
                "home_rank_display": (
                    f"#{home_rank}" if home_rank is not None else "NR"
                ),
                "away_rank_score": round(away_rank_score, 2),
                "home_rank_score": round(home_rank_score, 2),
                "ranking_quality": round(ranking_quality, 2),
                "spread": spread,
                "absolute_spread": abs(spread) if spread is not None else None,
                "spread_score": (
                    round(spread_score, 2)
                    if spread_score is not None
                    else None
                ),
                "watchability_index": (
                    round(watchability, 2)
                    if watchability is not None
                    else None
                ),
                "data_updated_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        return dataframe

    dataframe["weekly_watchability_rank"] = (
        dataframe["watchability_index"]
        .rank(method="min", ascending=False, na_option="bottom")
        .astype("Int64")
    )

    return dataframe.sort_values(
        ["week", "weekly_watchability_rank", "start_date"],
        na_position="last",
    )


def determine_season() -> int:
    """
    Use the current year.

    This works during the college football season and can be overridden
    in GitHub Actions with the CFB_SEASON environment variable.
    """

    override = os.environ.get("CFB_SEASON")

    if override:
        return int(override)

    return datetime.now(timezone.utc).year


def determine_weeks() -> list[int]:
    """
    Update all regular-season weeks.

    Rebuilding every week prevents older weeks from disappearing and
    allows revised lines or rankings to be captured.
    """

    configured_weeks = os.environ.get("CFB_WEEKS", "0-15")

    if "-" in configured_weeks:
        first, last = configured_weeks.split("-", maxsplit=1)
        return list(range(int(first), int(last) + 1))

    return [
        int(value.strip())
        for value in configured_weeks.split(",")
        if value.strip()
    ]


def main() -> None:
    season = determine_season()
    weeks = determine_weeks()

    all_weeks: list[pd.DataFrame] = []

    for week in weeks:
        print(f"Retrieving {season} regular-season week {week}...")

        try:
            weekly_data = build_week(season, week)
        except requests.HTTPError as error:
            print(f"Week {week} API request failed: {error}")
            continue

        if not weekly_data.empty:
            all_weeks.append(weekly_data)

    if not all_weeks:
        raise RuntimeError(
            f"No game data was returned for the {season} season."
        )

    final_data = pd.concat(all_weeks, ignore_index=True)

    final_data = final_data.drop_duplicates(
        subset=["season", "season_type", "week", "game_id"],
        keep="last",
    )

    final_data = final_data.sort_values(
        ["week", "weekly_watchability_rank", "start_date"],
        na_position="last",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_data.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(final_data):,} games to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
