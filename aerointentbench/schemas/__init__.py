"""Typed benchmark schemas and their JSON loading/validation.

Responsibility
--------------
Define the immutable domain models that every other layer speaks in, and own *all*
JSON parsing and validation for them. Nothing outside this package should call
``json.load`` on a benchmark specification file.

Planned modules (added in ``feature/v1-schemas``)
------------------------------------------------
- ``contract``       -- mission contract: what must be achieved, and the hard constraints.
- ``episode``        -- initial episode state: platform, path, trace, allowed configs, seed.
- ``configuration``  -- configuration identity (``config_id``/``model_id``/``strategy``)
                        and the configuration catalog.
- ``platform``       -- UAV platform profile: capacity, flight power, comms energy.
- ``task_spec``      -- task definition: evidence type, matching rule, deduplication.
- ``runtime_state``  -- the policy-visible observation built fresh at every decision step.
- ``loading``        -- schema-version gate, field validation, and the single place a
                        future migration layer would hook in.

Boundaries
----------
- Models are frozen dataclasses; the simulator never mutates a schema object in place.
- ``RuntimeState`` must never carry ground truth or future network values. See
  ``docs/architecture.md`` ("Policy-visible vs hidden state").
- Validation is strict: unknown fields are rejected, not silently ignored, so that a
  misspelled key never reads as an intentional default.
"""
