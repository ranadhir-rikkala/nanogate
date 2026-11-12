import pytest

from nanogate.dlp import Scanner, new_entities, normalize
from nanogate.models import DataClass

scanner = Scanner()


@pytest.mark.parametrize(
    "text, entity, data_class",
    [
        ("mail jane.doe@acme.com about it", "EMAIL", DataClass.CONFIDENTIAL),
        ("call (408) 555-0199 tomorrow", "PHONE", DataClass.CONFIDENTIAL),
        ("badge EMP-104233 is locked", "EMPLOYEE_ID", DataClass.CONFIDENTIAL),
        ("ssn 123-45-6789 on file", "SSN", DataClass.RESTRICTED),
        ("card 4111 1111 1111 1111 declined", "CREDIT_CARD", DataClass.RESTRICTED),
        ("host 10.20.30.40 is down", "IP_ADDRESS", DataClass.INTERNAL),
        ("key sk-abcdefghijklmnopqrstuvwx1234", "OPENAI_KEY", DataClass.SECRET),
        ("creds AKIAIOSFODNN7EXAMPLE leaked", "AWS_ACCESS_KEY", DataClass.SECRET),
        ("token ghp_" + "a1B2" * 9, "GITHUB_TOKEN", DataClass.SECRET),
        ("-----BEGIN RSA PRIVATE KEY-----\nMIIE", "PRIVATE_KEY", DataClass.SECRET),
        ("the db password is Tr0ub4dor&3xyz", "CREDENTIAL_ASSIGNMENT", DataClass.SECRET),
    ],
)
def test_detects_entity(text, entity, data_class):
    result = scanner.scan(text)
    assert result.entities.get(entity) == 1
    assert result.data_class == data_class
    assert result.has_secret == (data_class == DataClass.SECRET)


@pytest.mark.parametrize(
    "text",
    [
        "How do I reset the VPN client on a laptop?",
        "order 1234 5678 9012 3456 has shipped",
        "ticket 000-12-3456 and 666-12-3456 are not SSNs",
        "version 1.2.3.4000 of the agent",
        "what is a good password policy for contractors?",
        "the invoice total was 1234567890123",
    ],
)
def test_clean_text_stays_public(text):
    result = scanner.scan(text)
    assert result.entities == {}
    assert result.data_class == DataClass.PUBLIC


@pytest.mark.parametrize(
    "text, entity",
    [
        ("reach me at jane dot doe at acme dot com", "EMAIL"),
        ("jane [at] acme [dot] com", "EMAIL"),
        ("ssn 1 2 3 4 5 6 7 8 9 0 1 1 1 1 1 1".replace("1 2 3 4 5 6 7 8 9 0 1 1 1 1 1 1", "4 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"), "CREDIT_CARD"),
        ("sk-abcdefgh​ijklmnopqrstuvwx1234", "OPENAI_KEY"),
        ("ssn １２３-４５-６７８９", "SSN"),
    ],
)
def test_obfuscated_values_are_still_found(text, entity):
    assert entity in scanner.scan(text).entities


def test_result_never_contains_matched_text():
    result = scanner.scan("jane.doe@acme.com / 123-45-6789")
    rendered = repr(result.summary()) + repr(result.findings)
    assert "jane" not in rendered and "6789" not in rendered


def test_redact_replaces_values_with_entity_labels():
    text, count = scanner.redact("email jane.doe@acme.com or call 408-555-0199")
    assert text == "email [EMAIL] or call [PHONE]"
    assert count == 2
    assert scanner.scan(text).entities == {}


def test_redact_handles_obfuscated_input():
    text, count = scanner.redact("write to jane [at] acme [dot] com please")
    assert "[EMAIL]" in text and "acme" not in text
    assert count == 1


def test_redact_can_target_specific_entities():
    text, count = scanner.redact("jane@acme.com from 10.0.0.1", only=frozenset({"EMAIL"}))
    assert text == "[EMAIL] from 10.0.0.1"
    assert count == 1


def test_new_entities_reports_only_what_the_input_lacked():
    source = scanner.scan("my email is jane@acme.com")
    output = scanner.scan("jane@acme.com and bob@acme.com, ssn 123-45-6789")
    assert new_entities(output, source) == {"EMAIL": 1, "SSN": 1}
    assert new_entities(source, source) == {}


def test_custom_class_map_and_floor():
    custom = Scanner(classes={"EMAIL": DataClass.RESTRICTED}, floor=DataClass.INTERNAL)
    assert custom.scan("hello").data_class == DataClass.INTERNAL
    assert custom.scan("a@b.io").data_class == DataClass.RESTRICTED


def test_normalize_is_stable_on_plain_text():
    text = "Look at the dashboard and dot the i's."
    assert normalize(text) == text
