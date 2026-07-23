# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

PLEX = {
    "request_id": "request",
    "principal_id": "tenant",
    "group_id": "workflow",
    "generation_id": 1,
    "terminal": False,
}


def test_chat_request_forwards_plex_metadata():
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=8,
        plex=PLEX,
    )

    sampling_params = request.to_sampling_params(
        max_tokens=8,
        default_sampling_params={},
    )

    assert sampling_params.extra_args == {"plex": PLEX}


def test_completion_request_forwards_plex_metadata():
    request = CompletionRequest(
        model="test-model",
        prompt="Hello",
        max_tokens=8,
        plex=PLEX,
    )

    sampling_params = request.to_sampling_params(
        max_tokens=8,
        default_sampling_params={},
    )

    assert sampling_params.extra_args == {"plex": PLEX}


def test_responses_request_forwards_plex_metadata():
    request = ResponsesRequest(
        model="test-model",
        input="Hello",
        max_output_tokens=8,
        plex=PLEX,
    )

    sampling_params = request.to_sampling_params(default_max_tokens=8)

    assert sampling_params.extra_args == {"plex": PLEX}
