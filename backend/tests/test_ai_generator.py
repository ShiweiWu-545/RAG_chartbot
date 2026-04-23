"""
Black-box tests for AI Generator response orchestration.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from ai_generator import AIGenerator


def make_tool_call(
    call_id="call_123", name="search_course_content", arguments='{"query": "Python basics"}'
):
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = name
    tool_call.function.arguments = arguments
    return tool_call


def make_response(content="", tool_calls=None, finish_reason="stop"):
    response = MagicMock()
    response.choices = [MagicMock()]
    choice = response.choices[0]
    choice.finish_reason = finish_reason
    choice.message.content = content
    if tool_calls is not None:
        choice.message.tool_calls = tool_calls
        choice.finish_reason = "tool_calls"
    return response


class TestAIGenerator:
    @pytest.fixture
    def ai_generator(self):
        with patch("ai_generator.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            generator = AIGenerator(
                api_key="test_key", model="test-model", base_url="https://test.api"
            )
            generator.client = mock_client
            return generator

    def test_generate_response_without_tools_returns_single_api_answer(self, ai_generator):
        first_response = make_response(content="This is a test response from the AI.")
        ai_generator.client.chat.completions.create.return_value = first_response

        result = ai_generator.generate_response(
            query="What is Python?", tools=None, tool_manager=None
        )

        assert result == "This is a test response from the AI."
        assert ai_generator.client.chat.completions.create.call_count == 1

        call_kwargs = ai_generator.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][1]["role"] == "user"
        assert call_kwargs["messages"][1]["content"] == "What is Python?"
        assert "tool_choice" not in call_kwargs

    def test_generate_response_two_tool_rounds_uses_previous_results(self, ai_generator):
        first_response = make_response(
            content="I will search for the lesson topic first.",
            tool_calls=[make_tool_call(arguments='{"query": "course outline for course X"}')],
        )
        second_response = make_response(
            content="I found the lesson topic, now I will search for a matching course.",
            tool_calls=[
                make_tool_call(
                    call_id="call_456", arguments='{"query": "course with distributed systems"}'
                )
            ],
        )
        third_response = make_response(
            content="Course Y covers the same topic as lesson 4 of course X."
        )
        ai_generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.get_tool_allowed_arguments.return_value = {"query"}
        mock_tool_manager.execute_tool.side_effect = [
            "Lesson 4 title: Distributed Systems",
            "Course Y result: Distributed Systems",
        ]

        tools = [{"type": "function", "function": {"name": "search_course_content"}}]
        result = ai_generator.generate_response(
            query="Search for a course that discusses the same topic as lesson 4 of course X",
            tools=tools,
            tool_manager=mock_tool_manager,
        )

        assert result == "Course Y covers the same topic as lesson 4 of course X."
        assert ai_generator.client.chat.completions.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

        first_call = ai_generator.client.chat.completions.create.call_args_list[0].kwargs
        second_call = ai_generator.client.chat.completions.create.call_args_list[1].kwargs
        third_call = ai_generator.client.chat.completions.create.call_args_list[2].kwargs

        assert first_call["tools"] == tools
        assert first_call["tool_choice"] == {"type": "auto"}
        assert second_call["tools"] == tools
        assert second_call["tool_choice"] == {"type": "auto"}
        assert third_call["tools"] == tools
        assert third_call["tool_choice"] == {"type": "auto"}

        second_messages = second_call["messages"]
        assert second_messages[0]["role"] == "system"
        assert second_messages[1]["role"] == "user"
        assert second_messages[2]["role"] == "assistant"
        assert second_messages[3]["role"] == "tool"
        assert second_messages[3]["content"] == "Lesson 4 title: Distributed Systems"

        third_messages = third_call["messages"]
        assert third_messages[-1]["role"] == "tool"
        assert third_messages[-1]["content"] == "Course Y result: Distributed Systems"

        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="course outline for course X"
        )

    def test_generate_response_stops_after_two_tool_rounds(self, ai_generator):
        first_response = make_response(
            content="First search underway.",
            tool_calls=[
                make_tool_call(
                    call_id="call_1", arguments='{"query": "course outline for course X"}'
                )
            ],
        )
        second_response = make_response(
            content="Second search underway.",
            tool_calls=[make_tool_call(call_id="call_2", arguments='{"query": "lesson 4 topic"}')],
        )
        third_response = make_response(
            content="I need one more search, but the limit should stop here.",
            tool_calls=[make_tool_call(call_id="call_3", arguments='{"query": "extra follow up"}')],
        )
        ai_generator.client.chat.completions.create.side_effect = [
            first_response,
            second_response,
            third_response,
        ]

        mock_tool_manager = MagicMock()
        mock_tool_manager.get_tool_allowed_arguments.return_value = {"query"}
        mock_tool_manager.execute_tool.side_effect = [
            "Lesson 4 title: Distributed Systems",
            "Matching course: Course Y",
        ]

        result = ai_generator.generate_response(
            query="Find a course related to lesson 4 of course X",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=mock_tool_manager,
        )

        assert result == "I need one more search, but the limit should stop here."
        assert ai_generator.client.chat.completions.create.call_count == 3
        assert mock_tool_manager.execute_tool.call_count == 2

    def test_generate_response_handles_tool_execution_errors_gracefully(self, ai_generator):
        first_response = make_response(
            content="I will look up the course outline first.",
            tool_calls=[make_tool_call(arguments='{"query": "course outline for course X"}')],
        )
        ai_generator.client.chat.completions.create.return_value = first_response

        mock_tool_manager = MagicMock()
        mock_tool_manager.get_tool_allowed_arguments.return_value = {"query"}
        mock_tool_manager.execute_tool.side_effect = RuntimeError("backend exploded")

        result = ai_generator.generate_response(
            query="Find lesson 4 topic",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Tool execution failed: backend exploded"
        assert ai_generator.client.chat.completions.create.call_count == 1
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="course outline for course X"
        )

    def test_generate_response_filters_tool_arguments_and_keeps_null_values(self, ai_generator):
        first_response = make_response(
            content="I will use the tool now.",
            tool_calls=[
                make_tool_call(
                    arguments='{"query": "lesson summary", "course_name": null, "ignored": "value"}'
                )
            ],
        )
        second_response = make_response(content="Final answer")
        ai_generator.client.chat.completions.create.side_effect = [first_response, second_response]

        mock_tool_manager = MagicMock()
        mock_tool_manager.get_tool_allowed_arguments.return_value = {"query", "course_name"}
        mock_tool_manager.execute_tool.return_value = "Lesson summary content"

        result = ai_generator.generate_response(
            query="Summarize lesson 1",
            tools=[{"type": "function", "function": {"name": "search_course_content"}}],
            tool_manager=mock_tool_manager,
        )

        assert result == "Final answer"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="lesson summary", course_name=None
        )

    def test_conversation_history_included(self, ai_generator):
        ai_generator.client.chat.completions.create.return_value = make_response(content="Done")

        history = "User: What is Python?\nAssistant: Python is a programming language."

        ai_generator.generate_response(
            query="Tell me more", conversation_history=history, tools=None, tool_manager=None
        )

        call_kwargs = ai_generator.client.chat.completions.create.call_args.kwargs
        system_msg = call_kwargs["messages"][0]["content"]

        assert "Previous conversation" in system_msg
        assert "Python is a programming language" in system_msg

    def test_tool_choice_auto_when_tools_provided(self, ai_generator):
        ai_generator.client.chat.completions.create.return_value = make_response(
            content="No tool needed"
        )

        ai_generator.generate_response(
            query="Test", tools=[{"type": "function"}], tool_manager=None
        )

        call_kwargs = ai_generator.client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == {"type": "auto"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
