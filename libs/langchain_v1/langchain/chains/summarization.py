"""Inline Summarization Chain

TODO(Eugene): see if we can add annotations / citations.
"""

from __future__ import annotations

from typing import NotRequired, cast

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, MessageLikeRepresentation
from langgraph.graph import END, StateGraph
from langgraph.pregel import Pregel
from typing_extensions import TypedDict

from langchain._internal.utils import RunnableCallable


class InlineSummarizationState(TypedDict):
    """State for inline summarization."""

    documents: list[Document]
    """List of documents to summarize."""
    summary: NotRequired[str]
    """Summary of the documents, available after summarization."""


class InputSchema(TypedDict):
    """Input for the inline summarization chain."""

    documents: list[Document]
    """List of documents to summarize."""


class OutputSchema(TypedDict):
    """Output of the inline summarization chain."""

    summary: str
    """Summary of the documents."""


class SummarizationNodeUpdate(TypedDict):
    """Update for the summarization node."""

    summary: str
    """Summary of the documents."""


class InlineSummarizer:
    def __init__(self, model: BaseChatModel, prompt: str | None) -> None:
        """Initialize the InlineSummarizer with a chat model."""
        self.model = model
        self.prompt = prompt

    def _get_prompt(
        self, state: InlineSummarizationState
    ) -> list[MessageLikeRepresentation]:
        """Generates the prompt for inline summarization."""
        if self.prompt is None:
            default_system = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that summarizes text. "
                        "Please provide a concise summary of the documents "
                        "provided by the user."
                    ),
                }
            ]
        elif isinstance(self.prompt, str):
            default_system = [{"role": "system", "content": self.prompt}]
        else:
            msg = f"Invalid prompt type: {type(self.prompt)}. Expected str or None."
            raise TypeError(msg)

        inlined_docs = "---\n\n".join(doc.page_content for doc in state["documents"])
        return [
            default_system,
            {
                "role": "user",
                "content": inlined_docs,
            },
        ]

    def create_summarization_node(
        self,
    ) -> RunnableCallable[InlineSummarizationState, SummarizationNodeUpdate]:
        """Creates a node for inline summarization."""

        def _summarize_node(state: InlineSummarizationState) -> SummarizationNodeUpdate:
            """Builds a LangGraph for inline summarization."""
            prompt = self._get_prompt(state)
            response = cast("AIMessage", self.model.invoke(prompt))
            return {"summary": response.text()}

        async def _asummarize_node(
            state: InlineSummarizationState,
        ) -> SummarizationNodeUpdate:
            """Asynchronous version of the summarize node."""
            prompt = self._get_prompt(state)
            response = cast("AIMessage", await self.model.ainvoke(prompt))
            return {"summary": response.text()}

        return RunnableCallable[InlineSummarizationState, SummarizationNodeUpdate](
            _summarize_node,
            _asummarize_node,
        )

    def build(self) -> Pregel:
        """Builds the LangGraph for inline summarization."""
        builder = StateGraph(
            InlineSummarizationState,
            input_schema=InputSchema,
            output_schema=OutputSchema,
        )
        builder.add_node("summarize", self.create_summarization_node())
        builder.set_entry_point("summarize")
        builder.add_edge("summarize", END)
        return builder


__all__ = ["InlineSummarizer"]
