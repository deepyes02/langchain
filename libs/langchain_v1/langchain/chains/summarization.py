"""Inline Summarization Chain"""

from typing import NotRequired, cast
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.pregel import Pregel

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import MessageLikeRepresentation, AIMessage


class InlineSummarizationState(TypedDict):
    """State for inline summarization."""

    documents: list[Document]
    summary: NotRequired[str]


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

    def _summarize_node(self, state: InlineSummarizationState) -> TypedDict(
        "Update", {"summary": str}
    ):
        """Builds a LangGraph for inline summarization."""
        prompt = self._get_prompt(state)
        response = cast(AIMessage, self.model.invoke(prompt))
        return {"summary": response.text()}

    async def _asummarize_node(self, state: InlineSummarizationState) -> TypedDict(
        "Update", {"summary": str}
    ):
        """Asynchronous version of the summarize node."""
        prompt = self._get_prompt(state)
        response = cast(AIMessage, await self.model.ainvoke(prompt))
        return {"summary": response.text()}

    def build(self) -> Pregel:
        """Builds the LangGraph for inline summarization."""
        builder = StateGraph(InlineSummarizationState)
        builder.add_node("summarize_inline", self._summarize_node)
        builder.set_entry_point("summarize_inline")
        builder.add_edge("summarize_inline", END)
        return builder


__all__ = ["InlineSummarizer"]
