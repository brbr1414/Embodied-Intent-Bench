"""Validation behaviour of the centralised loader.

These tests pin the two properties the rest of the benchmark relies on: a document is
either fully conforming or rejected with a locating message, and an unsupported schema
version is never silently reinterpreted.
"""

from __future__ import annotations

from typing import Any

import pytest

from aerointentbench.schemas.contract import load_contract
from aerointentbench.schemas.loading import (
    SUPPORTED_SCHEMA_VERSIONS,
    DocumentReader,
    SchemaValidationError,
    SchemaVersionError,
    open_document,
)


# --- schema version gate ---------------------------------------------------------


def test_v1_supports_exactly_one_schema_version() -> None:
    assert SUPPORTED_SCHEMA_VERSIONS == frozenset({"1.0"})


def test_missing_schema_version_is_rejected(write_json, valid_contract_payload) -> None:
    del valid_contract_payload["schema_version"]
    with pytest.raises(SchemaVersionError, match="missing required field 'schema_version'"):
        load_contract(write_json(valid_contract_payload))


@pytest.mark.parametrize("version", ["0.9", "1.1", "2.0", "1", ""])
def test_unsupported_schema_version_is_rejected(
    write_json, valid_contract_payload, version: str
) -> None:
    valid_contract_payload["schema_version"] = version
    with pytest.raises(SchemaVersionError, match="unsupported schema_version"):
        load_contract(write_json(valid_contract_payload))


def test_non_string_schema_version_is_rejected(write_json, valid_contract_payload) -> None:
    valid_contract_payload["schema_version"] = 1.0
    with pytest.raises(SchemaVersionError, match="must be a string"):
        load_contract(write_json(valid_contract_payload))


# --- strictness ------------------------------------------------------------------


def test_unknown_field_is_rejected_rather_than_ignored(
    write_json, valid_contract_payload
) -> None:
    """A typo must fail loudly instead of silently reading as an absent field."""
    valid_contract_payload["deadline_sec"] = 60.0
    with pytest.raises(SchemaValidationError, match=r"unknown field\(s\) \['deadline_sec'\]"):
        load_contract(write_json(valid_contract_payload))


def test_missing_required_field_names_the_field(write_json, valid_contract_payload) -> None:
    del valid_contract_payload["deadline_s"]
    with pytest.raises(SchemaValidationError, match="missing required field 'deadline_s'"):
        load_contract(write_json(valid_contract_payload))


def test_error_message_locates_the_document_and_schema(
    write_json, valid_contract_payload
) -> None:
    del valid_contract_payload["task_id"]
    path = write_json(valid_contract_payload, name="broken_contract.json")
    with pytest.raises(SchemaValidationError) as error:
        load_contract(path)
    assert "broken_contract.json -> Contract" in str(error.value)


# --- file-level failures ---------------------------------------------------------


def test_invalid_json_is_reported_as_a_schema_error(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SchemaValidationError, match="invalid JSON"):
        load_contract(path)


def test_non_object_top_level_is_rejected(write_json) -> None:
    with pytest.raises(SchemaValidationError, match="top level must be a JSON object"):
        load_contract(write_json([1, 2, 3]))


def test_missing_file_is_reported_as_a_schema_error(tmp_path) -> None:
    with pytest.raises(SchemaValidationError, match="cannot be read"):
        load_contract(tmp_path / "absent.json")


# --- field readers ---------------------------------------------------------------


def _reader(data: dict[str, Any]) -> DocumentReader:
    return DocumentReader(data, context="test", allowed_fields=data.keys())


def test_get_int_rejects_bool() -> None:
    """bool is a subclass of int; accepting it would turn ``true`` into ``1``."""
    with pytest.raises(SchemaValidationError, match="must be an integer"):
        _reader({"seed": True}).get_int("seed")


def test_get_float_rejects_bool_and_nan() -> None:
    with pytest.raises(SchemaValidationError, match="must be a number"):
        _reader({"x": False}).get_float("x")
    with pytest.raises(SchemaValidationError, match="must not be NaN"):
        _reader({"x": float("nan")}).get_float("x")


def test_get_float_accepts_an_integer_json_value() -> None:
    assert _reader({"x": 20}).get_float("x") == 20.0


@pytest.mark.parametrize(
    ("bounds", "value", "expected_message"),
    [
        ({"minimum": 0.0}, -0.1, "must be >= 0.0"),
        ({"maximum": 1.0}, 1.1, "must be <= 1.0"),
        ({"exclusive_minimum": 0.0}, 0.0, "must be > 0.0"),
    ],
)
def test_get_float_enforces_bounds(
    bounds: dict[str, float], value: float, expected_message: str
) -> None:
    with pytest.raises(SchemaValidationError, match=expected_message):
        _reader({"x": value}).get_float("x", **bounds)


def test_get_fraction_accepts_the_closed_unit_interval() -> None:
    assert _reader({"x": 0.0}).get_fraction("x") == 0.0
    assert _reader({"x": 1.0}).get_fraction("x") == 1.0


def test_get_str_rejects_blank_values() -> None:
    with pytest.raises(SchemaValidationError, match="must not be empty"):
        _reader({"x": "   "}).get_str("x")


def test_get_optional_str_treats_null_and_absent_alike() -> None:
    assert _reader({"x": None}).get_optional_str("x") is None
    assert _reader({}).get_optional_str("x") is None
    assert _reader({"x": "value"}).get_optional_str("x") == "value"


def test_get_str_tuple_rejects_duplicates_and_empties() -> None:
    with pytest.raises(SchemaValidationError, match=r"duplicate entries \['A'\]"):
        _reader({"x": ["A", "B", "A"]}).get_str_tuple("x")
    with pytest.raises(SchemaValidationError, match="must not be empty"):
        _reader({"x": []}).get_str_tuple("x")
    with pytest.raises(SchemaValidationError, match=r"field 'x'\[1\] must be a non-empty string"):
        _reader({"x": ["A", 2]}).get_str_tuple("x")


def test_get_str_tuple_rejects_a_bare_string() -> None:
    """A string is a Sequence; accepting one would read "ABC" as three entries."""
    with pytest.raises(SchemaValidationError, match="must be a list of strings"):
        _reader({"x": "ABC"}).get_str_tuple("x")


def test_get_scalar_mapping_permits_scalars_and_rejects_structure() -> None:
    reader = _reader({"parameters": {"pruning_ratio": 0.5, "enabled": True, "note": "x"}})
    assert reader.get_scalar_mapping("parameters") == {
        "pruning_ratio": 0.5,
        "enabled": True,
        "note": "x",
    }
    assert _reader({}).get_scalar_mapping("parameters") == {}
    with pytest.raises(SchemaValidationError, match="must be a JSON scalar"):
        _reader({"parameters": {"nested": {"a": 1}}}).get_scalar_mapping("parameters")


def test_open_document_rejects_unknown_nested_fields(write_json, valid_trace_payload) -> None:
    valid_trace_payload["segments"][0]["jitter_ms"] = 5.0
    reader = open_document(
        write_json(valid_trace_payload),
        document_type="NetworkTrace",
        allowed_fields=("trace_id", "segments"),
    )
    with pytest.raises(SchemaValidationError, match=r"unknown field\(s\) \['jitter_ms'\]"):
        reader.get_object_list(
            "segments",
            allowed_fields=("start_s", "end_s", "bandwidth_mbps", "rtt_ms", "packet_loss_frac"),
        )
