"""A local Guardrails AI validator for tests (no Hub download, no network)."""

from guardrails.validators import FailResult, PassResult, Validator, register_validator


@register_validator(name="agent-svc/no-forbidden-word", data_type="string")
class NoForbiddenWord(Validator):
    def _validate(self, value, metadata):  # type: ignore[no-untyped-def]
        if "forbidden" in value.lower():
            return FailResult(error_message="contains a forbidden word")
        return PassResult()

    def validate(self, value, metadata):  # type: ignore[no-untyped-def]
        return self._validate(value, metadata)
