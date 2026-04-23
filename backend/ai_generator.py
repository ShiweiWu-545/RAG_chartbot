import json
import logging
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)


class AIGenerator:
    """Handles interactions with MiniMax API (OpenAI-compatible) for generating responses"""

    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may make up to 2 sequential tool rounds when a follow-up search is needed
- Each tool round should build on the previous tool result before deciding whether to search again
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- If a first search is insufficient, perform one follow-up search using the earlier result
- Stop after at most 2 tool rounds and provide the best direct answer available
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, base_url: str):
        self.client = OpenAI(
            api_key=api_key, base_url=base_url, http_client=httpx.Client(trust_env=False)
        )
        self.model = model

        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        system_content = self.SYSTEM_PROMPT
        if conversation_history:
            system_content = (
                f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]

        response = self._create_completion(messages, tools)

        tool_rounds = 0
        while tool_rounds < 2:
            choice = response.choices[0]
            tool_calls = self._extract_tool_calls(choice.message)

            if not tool_calls:
                return choice.message.content or ""

            if not tool_manager:
                logger.warning(
                    "Tool calls were returned but no tool manager was provided; returning assistant content only."
                )
                return choice.message.content or ""

            tool_call = tool_calls[0]
            if len(tool_calls) > 1:
                logger.warning(
                    "Multiple tool calls returned in one response for %s; executing only the first call.",
                    tool_call.function.name,
                )

            messages.append(
                self._build_assistant_tool_message(choice.message.content or "", tool_call)
            )

            try:
                allowed_args = None
                if hasattr(tool_manager, "get_tool_allowed_arguments"):
                    allowed_args = tool_manager.get_tool_allowed_arguments(tool_call.function.name)
                tool_args = self._parse_tool_arguments(
                    tool_call.function.name, tool_call.function.arguments, allowed_args
                )
                result = tool_manager.execute_tool(tool_call.function.name, **tool_args)
            except Exception as exc:
                logger.exception("Tool execution raised for %s", tool_call.function.name)
                return f"Tool execution failed: {exc}"

            result_text = str(result)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})

            if self._is_tool_failure(result_text):
                logger.warning(
                    "Tool execution failed for %s: %s", tool_call.function.name, result_text
                )
                return result_text

            tool_rounds += 1
            response = self._create_completion(messages, tools)

        return response.choices[0].message.content or ""

    def _create_completion(self, messages: List[Dict[str, Any]], tools: Optional[List] = None):
        api_params = {**self.base_params, "messages": messages}

        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.chat.completions.create(**api_params)

    def _build_assistant_tool_message(self, content: str, tool_call) -> Dict[str, Any]:
        return {
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        }

    def _extract_tool_calls(self, message) -> List[Any]:
        tool_calls = getattr(message, "__dict__", {}).get("tool_calls")
        if not tool_calls:
            return []
        if isinstance(tool_calls, list):
            return tool_calls
        if isinstance(tool_calls, tuple):
            return list(tool_calls)
        return [tool_calls]

    def _is_tool_failure(self, result_text: str) -> bool:
        failure_prefixes = ("Tool execution failed:", "Tool '", "Search error:")
        return isinstance(result_text, str) and result_text.startswith(failure_prefixes)

    def _parse_tool_arguments(
        self, tool_name: str, raw_arguments: Optional[str], allowed_args: Optional[set[str]] = None
    ) -> Dict[str, Any]:
        """Parse OpenAI-compatible JSON tool arguments and drop unknown fields."""
        try:
            parsed_args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON tool arguments for %s: %s", tool_name, raw_arguments)
            raise ValueError(f"Invalid tool arguments for '{tool_name}'") from exc

        if not isinstance(parsed_args, dict):
            logger.warning(
                "Tool arguments for %s must be a JSON object: %r", tool_name, parsed_args
            )
            raise ValueError(f"Invalid tool arguments for '{tool_name}'")

        if allowed_args is not None:
            parsed_args = {key: value for key, value in parsed_args.items() if key in allowed_args}

        return parsed_args
