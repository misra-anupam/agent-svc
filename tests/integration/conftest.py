import os

# Colima/rootless Docker cannot mount the socket into testcontainers' reaper container.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
