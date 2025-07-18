from types import SimpleNamespace

import apps.onnx_chat.main as chat


def test_classify(monkeypatch):
    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input")]

        def run(self, *_):
            return [[42]]

    monkeypatch.setattr(chat, "load_session", lambda m: FakeSession())
    assert chat.classify("hello", "model") == "42"

    monkeypatch.setattr(chat, "load_session", lambda m: None)
    assert chat.classify("it is good", "model") == "positive"
    assert chat.classify("bad text", "model") == "negative"



