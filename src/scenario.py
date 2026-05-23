"""Shared scenario picker for the multi-page Streamlit app."""

import streamlit as st

from src import queries


def scenario_sidebar(label: str = "Scenario", key: str = "scenario_choice") -> int:
    """Render the scenario selector and return the selected scenario_id.

    The selection persists across pages via `st.session_state[key]`, so a choice
    made on one page is the default when the user navigates to another.
    """
    scenarios = queries.get_scenarios()
    name_to_id = dict(zip(scenarios["name"], scenarios["id"]))
    names = list(name_to_id.keys())

    previous = st.session_state.get(key)
    default_index = names.index(previous) if previous in names else 0

    chosen = st.sidebar.selectbox(label, options=names, index=default_index, key=key)
    return int(name_to_id[chosen])
