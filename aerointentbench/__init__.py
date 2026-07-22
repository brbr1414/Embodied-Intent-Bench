"""AeroIntentBench: intent-conditioned, resource-aware inference configuration selection.

AeroIntentBench evaluates *policies that choose an inference configuration* during a
UAV mission. It does not evaluate path planning, navigation, flight control, or
open-ended language understanding.

A policy observes a structured mission contract plus a policy-visible runtime state,
then selects one configuration ID from a fixed pool. The runner executes (or replays)
that configuration, advances time, battery, network usage, path progress, and mission
evidence, and finally computes mission-level metrics.

See ``docs/v1_spec.md`` for the V1 scope and ``docs/architecture.md`` for the module
boundaries every contributor is expected to respect.
"""

__all__ = ["SCHEMA_VERSION", "__version__"]

__version__ = "0.1.0.dev0"

#: The only benchmark schema version supported by this release. Every specification
#: file (contract, episode, config catalog, platform, task spec, network trace) and
#: every emitted result file carries a ``schema_version`` field. Loading a file with
#: any other value must fail with a clear validation error rather than being coerced.
#: Migration support, when it arrives, belongs in ``aerointentbench.schemas.loading``.
SCHEMA_VERSION = "1.0"
