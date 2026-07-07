from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .documents import SearchResult
from .embeddings import SentenceTransformerEmbedder
from .llm import GroqChatClient
from .moderation import validate_question
from .prompting import build_user_prompt
from .vectorstore import ChromaStore


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: list[SearchResult]
    model: str


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = SentenceTransformerEmbedder(settings.embedding_model)
        self.store = ChromaStore(settings.chroma_dir, settings.collection_name, self.embedder)

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        is_valid, reason = validate_question(question)
        if not is_valid:
            raise ValueError(reason or "Question invalide.")
        k = top_k or self.settings.top_k
        return self.store.search(question, top_k=k)

    def ask(self, question: str, top_k: int | None = None) -> RagAnswer:
        results = self.retrieve(question, top_k=top_k)
        if not results:
            return RagAnswer(
                question=question,
                answer="Je ne sais pas avec le corpus fourni.",
                sources=[],
                model=self.settings.groq_model,
            )

        prompt = build_user_prompt(question, results)
        llm = GroqChatClient(self.settings)
        llm_answer = llm.answer(prompt)
        return RagAnswer(
            question=question,
            answer=llm_answer.content,
            sources=results,
            model=llm_answer.model,
        )
