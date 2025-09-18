# SPDX-License-Identifier: Apache-2.0

import json
import re
from collections.abc import Sequence
from typing import Union

import partial_json_parser
from partial_json_parser.core.options import Allow

from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall, DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser,
    ToolParserManager,
)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import random_uuid

logger = init_logger(__name__)


@ToolParserManager.register_module("llama3_nemotron_json")  # vLLM v10 bug that validates tool-call-parser before injecting our one, so we override
class LlamaNemotronJSONToolParser(ToolParser):

    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        
        logger.info(f"LlamaNemotronJSONToolParser initialized with tokenizer: {tokenizer}")

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<TOOLCALL>"
        self.tool_call_end_token: str = "</TOOLCALL>"

        self.tool_call_regex = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        
        logger.info(f"Extracting tool calls from model output: {model_output}")

        if self.tool_call_start_token not in model_output:
            logger.info(f"No tool call start token found in model output")
            logger.info(f"Returning ExtractedToolCallInformation(tools_called=False, tool_calls=[], content={model_output})")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        else:

            try:
                logger.info(f"Tool call start token found in model output")
                str_tool_calls = self.tool_call_regex.findall(model_output)[0].strip()
                if not str_tool_calls.startswith("["):
                    str_tool_calls = "[" + str_tool_calls
                if not str_tool_calls.endswith("]"):
                    str_tool_calls = "]" + str_tool_calls
                json_tool_calls = json.loads(str_tool_calls)
                tool_calls = []
                for tool_call in json_tool_calls:
                    try:
                        logger.info(f"Adding tool call: {tool_call}")
                        tool_calls.append(ToolCall(
                            type="function",
                            function=FunctionCall(
                                name=tool_call["name"],
                                arguments=json.dumps(tool_call["arguments"], ensure_ascii=False) \
                                    if isinstance(tool_call["arguments"], dict) else tool_call["arguments"],
                            ),
                        ))
                    except:
                        logger.info(f"Error adding tool call: {tool_call}")
                        continue

                content = model_output[:model_output.rfind(self.tool_call_start_token)]
                logger.info(f"Content: {content}")
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )

            except Exception:
                logger.exception(f"Error in extracting tool call from response. Response: {model_output}")
                logger.info(f"Returning ExtractedToolCallInformation(tools_called=False, tool_calls=[], content={model_output})")
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output,
                )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:

        raise NotImplementedError("Tool calling is not supported in streaming mode!")
