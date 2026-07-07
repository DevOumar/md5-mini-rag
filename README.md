# MD5 Mini RAG - MD5 Mini RAG

Projet RAG pour interroger un corpus documentaire du corpus RAG avec citations de sources.

Le projet suit les pratiques vues en cours :

- indexation et interrogation separees ;
- index vectoriel persistant sur disque avec ChromaDB ;
- embeddings normalises et modele stocke avec la collection ;
- chunking lisible avec metadonnees de source ;
- prompt strict, reponse sourcee et fallback quand le contexte ne suffit pas ;
- tests et workflow Git avec commits progressifs.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
```

Renseigner ensuite `GROQ_API_KEY` dans `.env`.

## Corpus

Placer les documents sources dans `data/raw/`.

Formats supportes :

- `.txt`
- `.md`
- `.pdf`

Pour le rendu final, utiliser un corpus reel et tracer la provenance des fichiers dans `data/raw/README.md`.

## Indexation

```powershell
rag index
```

Cette commande cree ou met a jour l'index Chroma persistant dans `data/chroma/`.

## Interrogation

```powershell
rag ask "Comment s'appelle le chat bleu de Bob ?"
```

Pour tester seulement le retrieval avant le LLM :

```powershell
rag retrieve "Comment s'appelle le chat bleu de Bob ?"
```

## Verification

```powershell
pytest
```

## Regle importante

Ne pas reindexer a chaque question. L'indexation se fait avec `rag index`, puis les questions utilisent l'index persistant existant.
