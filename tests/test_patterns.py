from app.capture.synthetic import FAKE_AWS, FAKE_EMAIL, FAKE_GH, FAKE_JWT
from app.detection.patterns.secrets import find_matches, luhn_ok
from app.schema import Category


def rules(text):
    return {m.rule for m in find_matches(text)}


def test_known_formats():
    assert "aws_access_key" in rules(f"key={FAKE_AWS}")
    assert "github_token" in rules(FAKE_GH)
    assert "jwt" in rules(f"Authorization: {FAKE_JWT}")
    assert "private_key" in rules("-----BEGIN RSA PRIVATE KEY-----")


def test_pii_and_financial():
    assert "email" in rules(f"mail {FAKE_EMAIL}")
    assert "email" in rules("jane.doe @example.com")  # OCR often inserts a space
    assert "phone" in rules("call +1 415 555 0132")
    assert "card_number" in rules("card 4242 4242 4242 4242")
    assert "card_number" not in rules("order 1234 5678 9012 3456")  # fails Luhn
    assert luhn_ok("4242424242424242")


def test_assignment_and_entropy_for_unknown_formats():
    m = find_matches("password: Tr0ub4dor&3xplo!t")
    assert m and m[0].category == Category.CREDENTIALS and m[0].mandatory
    assert "high_entropy_token" in rules("x " + "aB3dE5fG7hJ9kL1mN3pQ5rS7tU9v")


def test_no_false_positives_on_prose():
    assert find_matches("Please review the draft before Friday. The offsite agenda: roadmap, hiring.") == []
    assert "high_entropy_token" not in rules("https://example.com/some/long/path/segment/here")
    assert "assigned_secret" not in rules("password: null")


def test_matches_never_carry_text():
    for m in find_matches(f"AWS {FAKE_AWS}"):
        assert not any(isinstance(v, str) and FAKE_AWS in v for v in vars(m).values())
