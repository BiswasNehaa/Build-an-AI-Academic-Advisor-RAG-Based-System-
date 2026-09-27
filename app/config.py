# =============================================================================
# config.py — Shared lightweight constants.
#
# Kept separate from advisor.py so values like MAX_CREDITS can be imported
# by app.py without triggering advisor.py's module-level model/index loading.
# =============================================================================

# Maximum credits a student can register per semester.
MAX_CREDITS = 30
