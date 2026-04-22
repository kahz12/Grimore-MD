"""
The Oracle: Retrieval-Augmented Generation (RAG) Engine.
This module combines semantic search (via the Connector) with LLM completion
(via the LLMRouter) to answer questions based on the vault's content.
"""
from pathlib import Path
from grimoire.cognition.llm_router import LLMRouter
from grimoire.cognition.embedder import Embedder
from grimoire.cognition.connector import Connector
from grimoire.memory.db import Database
from grimoire.utils.logger import get_logger
from grimoire.utils.security import SecurityGuard

logger = get_logger(__name__)

class Oracle:
    """
    Implements the RAG pipeline to provide context-aware answers to user queries.
    """
    def __init__(self, config, db: Database, router: LLMRouter, embedder: Embedder):
        self.config = config
        self.db = db
        self.router = router
        self.embedder = embedder
        self.connector = Connector(db, embedder)
        self.system_prompt_template = self._load_prompt()

    def _load_prompt(self):
        """Loads the Oracle's system prompt template."""
        prompt_path = Path(__file__).parent / "prompts" / "oracle.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        Main RAG entry point:
        1. Generates an embedding for the user's question.
        2. Retrieves the most relevant chunks from the database.
        3. Constructs a system prompt containing the retrieved context.
        4. Calls the LLM to generate a cited answer.
        """
        logger.info("oracle_query", question=question)
        
        # Step 1: Vectorize the question (BM25 alone is still useful if the
        # embedder is down, so we don't abort on a failed embed anymore).
        query_vector = self.embedder.embed(question)

        # Step 2: Retrieve relevant fragments (hybrid when possible).
        use_hybrid = (
            getattr(self.config.cognition, "hybrid_search", True)
            and self.db.fts_available
        )
        if use_hybrid:
            similar = self.connector.find_hybrid(
                query_text=question,
                query_vector=query_vector,
                top_k=top_k,
                rrf_k=getattr(self.config.cognition, "rrf_k", 60),
            )
        elif query_vector:
            similar = self.connector.find_similar_notes(query_vector, top_k=top_k)
        else:
            return {"answer": "I could not understand the question's essence.", "sources": []}

        if not similar:
            return {"answer": "Your vault seems empty of relevant whispers on this subject.", "sources": []}

        # Step 3: Format the context for the LLM
        context_parts = []
        sources = []
        for item in similar:
            # Fetch note title for citation
            with self.db._get_connection() as conn:
                row = conn.execute(
                    "SELECT title FROM notes WHERE id = ?",
                    (item['note_id'],),
                ).fetchone()
            if not row:
                # Handle potential orphan data
                logger.warning("orphan_embedding", note_id=item['note_id'])
                continue
            title = row[0]

            # Sanitize and wrap content to prevent prompt injection from notes
            safe_text = SecurityGuard.wrap_untrusted(
                SecurityGuard.sanitize_prompt(item['text']),
                label="source",
            )
            context_parts.append(f"--- Source: [[{title}]] ---\n{safe_text}")
            sources.append(title)

        if not context_parts:
            return {"answer": "Your vault seems empty of relevant whispers on this subject.", "sources": []}
        
        full_context = "\n\n".join(context_parts)
        
        # Step 4: LLM Generation
        # Inject retrieved context into the system prompt template
        system_prompt = self.system_prompt_template.replace("{context}", full_context)
        
        response = self.router.complete(
            prompt=f"Question: {question}",
            system_prompt=system_prompt,
        )

        if not response or not isinstance(response, dict):
            logger.warning("oracle_no_response")
            return {"answer": "The Oracle is silent.", "sources": list(set(sources))}

        return {
            "answer": response.get("answer", "The Oracle is silent."),
            "sources": list(set(sources))
        }
