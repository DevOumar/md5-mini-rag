from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .indexing import build_index
from .moderator import Moderator
from .prompting import format_source
from .rag import RAG

app = typer.Typer(
    help="Mini-RAG pedagogique avec ChromaDB, Sentence Transformers et Groq.",
    pretty_exceptions_show_locals=False,
)
console = Console()


@app.command()
def index(
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Supprime puis reconstruit la collection Chroma."),
    ] = False,
) -> None:
    """Indexe les documents de data/raw dans Chroma."""
    settings = get_settings()
    report = build_index(settings, reset=reset)
    console.print(
        f"[green]Index pret[/green] : {report.documents} documents, "
        f"{report.chunks} chunks, {report.indexed_chunks} chunks en base."
    )


@app.command()
def retrieve(
    question: Annotated[str, typer.Argument(help="Question a rechercher dans le corpus.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Nombre de chunks a retourner.")] = 4,
) -> None:
    """Teste seulement la recherche vectorielle."""
    settings = get_settings()
    service = RAG(settings)
    results = service.retrieve(question, top_k=top_k)
    _print_results(results)


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Question posee a l'assistant RAG.")],
    top_k: Annotated[int, typer.Option("--top-k", "-k", help="Nombre de chunks fournis au LLM.")] = 4,
) -> None:
    """Pose une question au RAG avec generation Groq."""
    settings = get_settings()
    service = RAG(settings)
    answer = service.ask(question, top_k=top_k)
    console.print(answer.answer)
    if answer.sources:
        console.print()
        _print_results(answer.sources)


@app.command()
def moderate(
    question: Annotated[str, typer.Argument(help="Question a analyser par l'agent moderateur.")],
) -> None:
    """Teste seulement l'agent moderateur Groq JSON."""
    settings = get_settings()
    result = Moderator(settings).moderate(question)
    console.print(
        {
            "prompt_injection": result.prompt_injection,
            "raison": result.raison,
        }
    )


@app.command()
def doctor() -> None:
    """Affiche la configuration chargee."""
    settings = get_settings()
    table = Table(title="Configuration")
    table.add_column("Cle")
    table.add_column("Valeur")
    table.add_row("raw_dir", str(settings.raw_dir))
    table.add_row("chroma_dir", str(settings.chroma_dir))
    table.add_row("collection_name", settings.collection_name)
    table.add_row("embedding_model", settings.embedding_model)
    table.add_row("groq_model", settings.groq_model)
    table.add_row("rag_prompt_path", str(settings.rag_prompt_path))
    table.add_row("moderator_prompt_path", str(settings.moderator_prompt_path))
    table.add_row("max_distance", str(settings.max_distance))
    table.add_row("groq_api_key", "present" if settings.groq_api_key else "absent")
    console.print(table)


def _print_results(results: list[object]) -> None:
    table = Table(title="Sources retrouvees")
    table.add_column("#", justify="right")
    table.add_column("Distance")
    table.add_column("Source")
    table.add_column("Extrait")

    for index, result in enumerate(results, start=1):
        distance = "" if result.distance is None else f"{result.distance:.4f}"
        excerpt = result.text.replace("\n", " ")
        if len(excerpt) > 180:
            excerpt = f"{excerpt[:177]}..."
        table.add_row(str(index), distance, format_source(result.metadata), excerpt)

    console.print(table)


if __name__ == "__main__":
    app()
