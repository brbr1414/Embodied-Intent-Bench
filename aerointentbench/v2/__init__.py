"""AeroIntentBench V2: the minimal *visual* closed loop.

V1 validated the benchmark semantics over an abstract mission -- profile replay,
mathematical state transitions, no imagery. V2 inserts a real visual world between the
mission state and the perception executor, as the deliberate bridge toward a future
physical (V3) simulator:

    2D aerial world (GeoTIFF / image)
      -> predefined UAV trajectory (position from mission time)
      -> position-dependent camera crop (windowed raster read)
      -> configuration-selection policy (the V1 Policy protocol, unchanged)
      -> image-based executor (prediction depends on RGB content)
      -> mission latency moves the UAV; busy executors skip observations
      -> capture-time ground truth scores the prediction
      -> mission-level evaluation (V1 empirical metric semantics)

Dependency boundary
-------------------
The V1 core stays runtime-dependency-free. Everything under this subpackage may use
``numpy``/``Pillow`` (and ``rasterio`` for GeoTIFF worlds), installed via the optional
``aerointentbench[v2]`` extra. Nothing outside ``aerointentbench.v2`` imports them.

Honesty
-------
V2 is low fidelity by design. Latency and energy are *simulated* (configured per
executor), the targets are *synthetic* rescue markers composited over real aerial
imagery, and results must not be described as realistic UAV perception performance.
V2 validates architecture, interfaces, closed-loop semantics, and mission-level
trade-offs -- not flight physics, hardware, or real-world model quality.
"""
