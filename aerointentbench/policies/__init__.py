"""Configuration-selection policies: the subject under evaluation.

Responsibility
--------------
Implement ``Policy.select_config(contract, state, configs) -> str``. A policy sees only
the contract, the policy-visible runtime state, and the allowed configuration
descriptions. Public configuration profiles are visible only when the benchmark mode
explicitly grants them -- never via an ad-hoc import.

Planned modules (added in ``feature/v1-policies``)
--------------------------------------------------
- ``base``           -- ``Policy`` protocol and the policy registry.
- ``static``         -- ``AlwaysLocalLightPolicy``, ``AlwaysLocalStrongPolicy``,
                        ``AlwaysRemoteStrongPolicy``.
- ``rule_based``     -- ``RuleBasedPolicy``: privacy, bandwidth, remaining comms budget,
                        battery headroom vs. required reserve, remaining deadline,
                        required quality threshold. Explicitly not claimed to be optimal.
- ``profile_greedy`` -- later; needs the public-profile visibility mode.

Boundaries
----------
A policy must never import a task evaluator, read ground truth, or hardcode fixture
config IDs as behaviour. Static baselines take their target config ID as a constructor
argument so they carry no fixture knowledge. Selection must be driven by typed
configuration metadata (``strategy.placement``, ``strategy.precision``, ...), not by
parsing ``config_id`` strings.
"""
