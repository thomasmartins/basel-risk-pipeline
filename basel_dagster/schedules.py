"""Schedules. Daily 06:00 full refresh is enough for Phase 1."""

from __future__ import annotations

from dagster import ScheduleDefinition

from basel_dagster.jobs import full_refresh_job

daily_full_refresh = ScheduleDefinition(
    name="daily_full_refresh",
    job=full_refresh_job,
    cron_schedule="0 6 * * *",
    execution_timezone="Europe/Lisbon",
)
