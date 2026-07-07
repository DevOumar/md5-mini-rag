from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .documents import SearchResult
from .embeddings import SentenceTransformerEmbedder
from .llm import GroqChatClient
from .moderator import Moderator
from .prompting import build_system_prompt, build_user_prompt
from .vectordb import VectorDB


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: list[SearchResult]
    model: str


class RAG:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = SentenceTransformerEmbedder(settings.embedding_model)
        self.store = VectorDB(settings.chroma_dir, settings.collection_name, self.embedder)
        self.moderator = Moderator(settings)

    def retrieve(self, question: str, top_k: int | None = None) -> list[SearchResult]:
        k = top_k or self.settings.top_k
        return self.store.search(question, top_k=k)

    def ask(self, question: str, top_k: int | None = None) -> RagAnswer:
        moderation = self.moderator.moderate(question)
        if moderation.prompt_injection:
            return RagAnswer(
                question=question,
                answer=f"Question bloquee par le moderateur : {moderation.raison}",
                sources=[],
                model=self.settings.groq_model,
            )

        results = self.retrieve(question, top_k=top_k)
        if not results:
            return RagAnswer(
                question=question,
                answer="Je ne sais pas avec le corpus fourni.",
                sources=[],
                model=self.settings.groq_model,
            )

        system_prompt = build_system_prompt(self.settings.rag_prompt_path, results)
        prompt = build_user_prompt(question, results)
        llm = GroqChatClient(self.settings)
        llm_answer = llm.answer(system_prompt, prompt)
        return RagAnswer(
            question=question,
            answer=llm_answer.content,
            sources=results,
            model=llm_answer.model,
        )
