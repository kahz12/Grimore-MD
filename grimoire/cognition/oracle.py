from pathlib import Path
from grimoire.cognition.llm_router import LLMRouter
from grimoire.cognition.embedder import Embedder
from grimoire.cognition.connector import Connector
from grimoire.memory.db import Database
from grimoire.utils.logger import get_logger
from grimoire.utils.security import SecurityGuard

logger = get_logger(__name__)

class Oracle:
    def __init__(self, config, db: Database, router: LLMRouter, embedder: Embedder):
        self.config = config
        self.db = db
        self.router = router
        self.embedder = embedder
        self.connector = Connector(db, embedder)
        self.system_prompt_template = self._load_prompt()

    def _load_prompt(self):
        prompt_path = Path(__file__).parent / "prompts" / "oracle.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()

    def ask(self, question: str, top_k: int = 5) -> dict:
        """
        Performs RAG: 
        1. Embeds question.
        2. Finds relevant chunks.
        3. Constructs prompt.
        4. Gets LLM answer.
        """
        logger.info("oracle_query", question=question)
        
        # 1. Get query vector
        query_vector = self.embedder.embed(question)
        if not query_vector:
            return {"answer": "I could not understand the question's essence.", "sources": []}

        # 2. Retrieve relevant chunks
        similar = self.connector.find_similar_notes(query_vector, top_k=top_k)
        
        if not similar:
            return {"answer": "Your vault seems empty of relevant whispers on this subject.", "sources": []}

        # 3. Build context
        context_parts = []
        sources = []
        for item in similar:
            with self.db._get_connection() as conn:
                row = conn.execute(
                    "SELECT title FROM notes WHERE id = ?",
                    (item['note_id'],),
                ).fetchone()
            if not row:
                # Orphan embedding: note was deleted but chunk lingers.
                logger.warning("orphan_embedding", note_id=item['note_id'])
                continue
            title = row[0]

            safe_text = SecurityGuard.wrap_untrusted(
                SecurityGuard.sanitize_prompt(item['text']),
                label="source",
            )
            context_parts.append(f"--- Source: [[{title}]] ---\n{safe_text}")
            sources.append(title)

        if not context_parts:
            return {"answer": "Your vault seems empty of relevant whispers on this subject.", "sources": []}
        
        full_context = "\n\n".join(context_parts)
        
        # 4. LLM Call
        system_prompt = self.system_prompt_template.replace("{context}", full_context)
        
        # Since Oracle response isn't necessarily JSON by default (it's a chat), 
        # we might need to adjust the router or use a different completion method.
        # But our router is set to 'format: json'. Let's adjust it for the Oracle.
        
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
