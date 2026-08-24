# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

from vllm_spark3_plugin import ARCHITECTURE, register
from vllm_spark3_plugin.spark3_tool_parser import Spark3ToolParser


def test_plugin_registers_model_and_tool_parser(monkeypatch):
    """The public plugin entry point registers Spark3's model and parser."""
    monkeypatch.setenv("SPARK3_PLUGIN_OVERRIDE", "1")

    register()

    assert ARCHITECTURE in ModelRegistry.get_supported_archs()
    assert ToolParserManager.get_tool_parser("spark") is Spark3ToolParser


def test_tool_parser_extracts_spark3_call():
    """The parser smoke test covers the wire format and typed arguments."""
    tool = ChatCompletionToolsParam(
        function={
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
            },
        }
    )
    parser = Spark3ToolParser(tokenizer=object(), tools=[tool])

    result = parser.extract_tool_calls(
        "Before "
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value>"
        "</tool_call> after",
        request=None,
    )

    assert result.tools_called
    assert result.content == "Before  after"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert result.tool_calls[0].function.arguments == '{"city":"Paris","days":3}'
