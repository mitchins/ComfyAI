import os
os.environ['UNIT_TEST_MODE'] = '1'

import openai_client
import image_utils
from vllm_query import VisionLLMQuery


def test_node_two_images(monkeypatch):
    called = {}
    def fake_chat(endpoint, model, messages, api_key=None, timeout=30):
        called['messages'] = messages
        return 'ok'
    monkeypatch.setattr(openai_client, 'chat_completion', fake_chat)
    monkeypatch.setattr(image_utils, 'image_to_base64', lambda x: 'imgdata')
    import vllm_query
    monkeypatch.setattr(vllm_query, 'image_to_base64', lambda x: 'imgdata')
    monkeypatch.setattr(vllm_query, 'chat_completion', fake_chat)

    node = VisionLLMQuery()
    result = node.run(
        api_endpoint='http://x',
        api_model='gpt',
        text_query='hi',
        image=object(),
        reference_image=object()
    )
    assert result[0] == 'ok'
    content = called['messages'][0]['content']
    assert len(content) == 3
    assert content[1]['type'] == 'image_url'
    assert content[2]['type'] == 'image_url'
