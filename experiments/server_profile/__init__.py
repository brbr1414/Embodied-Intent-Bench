"""Server-side execution profiling (development-machine stand-in server).

Responsibility and boundaries: measurement scripts and their outputs for the
server half of the benchmark's remote paths — full-model batch latency and split
tail latency. Nothing here runs in missions; the coming server model consumes the
resulting tables as labelled calibration data.
"""
