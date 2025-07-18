import pytest
from nodes.string_utils import fuzzy_match_bool

def test_fuzzy_match_bool_true_with_fuzz(monkeypatch):
    # Mock rapidfuzz to be available
    class MockFuzz:
        def ratio(self, s1, s2):
            if s1 == "yes" and s2 == "yes":
                return 100
            return 0
    monkeypatch.setattr("nodes.string_utils.fuzz", MockFuzz())
    assert fuzzy_match_bool("yes") is True

def test_fuzzy_match_bool_false_with_fuzz(monkeypatch):
    # Mock rapidfuzz to be available
    class MockFuzz:
        def ratio(self, s1, s2):
            if s1 == "no" and s2 == "no":
                return 100
            return 0
    monkeypatch.setattr("nodes.string_utils.fuzz", MockFuzz())
    assert fuzzy_match_bool("no") is False

def test_fuzzy_match_bool_none_with_fuzz(monkeypatch):
    # Mock rapidfuzz to be available
    class MockFuzz:
        def ratio(self, s1, s2):
            return 0
    monkeypatch.setattr("nodes.string_utils.fuzz", MockFuzz())
    assert fuzzy_match_bool("maybe") is None

def test_fuzzy_match_bool_true_no_fuzz(monkeypatch):
    # Mock rapidfuzz to be unavailable
    monkeypatch.setattr("nodes.string_utils.fuzz", None)
    assert fuzzy_match_bool("yes") is True
    assert fuzzy_match_bool("1") is True

def test_fuzzy_match_bool_false_no_fuzz(monkeypatch):
    # Mock rapidfuzz to be unavailable
    monkeypatch.setattr("nodes.string_utils.fuzz", None)
    assert fuzzy_match_bool("no") is False
    assert fuzzy_match_bool("0") is False

def test_fuzzy_match_bool_none_no_fuzz(monkeypatch):
    # Mock rapidfuzz to be unavailable
    monkeypatch.setattr("nodes.string_utils.fuzz", None)
    assert fuzzy_match_bool("maybe") is None

def test_fuzzy_match_bool_case_insensitivity(monkeypatch):
    monkeypatch.setattr("nodes.string_utils.fuzz", None)
    assert fuzzy_match_bool("Yes") is True
    assert fuzzy_match_bool("NO") is False

def test_fuzzy_match_bool_with_extra_words(monkeypatch):
    monkeypatch.setattr("nodes.string_utils.fuzz", None)
    assert fuzzy_match_bool("yes please") is True
    assert fuzzy_match_bool("no thank you") is False
