# 🧠 Project Grimoire v2.0 — Automated Knowledge Engine

> **Estado Actual:** ✅ MVP Completado (Fases 0-6). El sistema es totalmente funcional, autónomo y seguro.

---

## 1. Visión & Principios de Diseño

No es un script de etiquetado. Es un **daemon cognitivo** que vive en tu sistema, lee tu bóveda Markdown, construye un índice semántico vectorial, y descubre conexiones entre ideas dispares mientras tú haces otra cosa. *A second brain that runs on its own clock.*

**Principios cumplidos:**

1.  **Local-First, API-Optional.** Implementado vía Ollama. Privacidad garantizada.
2.  **Idempotente o muerto.** Hash-driven processing (SHA-256) funcional.
3.  **Non-destructive by default.** `git_guard` realiza commits automáticos pre-escritura. Modo `--dry-run` robusto.
4.  **Observable.** Logs estructurados con `structlog` y notificaciones Termux integradas.
5.  **Taxonomía controlada.** Normalización de tags en `taxonomy.py`.

---

## 2. Arquitectura en 4 Capas (Implementada)

```
┌─────────────────────────────────────────────────────────────┐
│  CAPA 1: SENSORIAL (Ingesta)                                │
│  watchdog → debounce (45s) → parser (frontmatter/body)      │
├─────────────────────────────────────────────────────────────┤
│  CAPA 2: COGNITIVA (Procesamiento)                          │
│  Tagger (LLM) → Embedder (Ollama/nomic) → Oracle (RAG)      │
├─────────────────────────────────────────────────────────────┤
│  CAPA 3: MEMORIA (Persistencia)                             │
│  SQLite (relacional) + Vector Store (BLOBs + Cosine Sim)    │
├─────────────────────────────────────────────────────────────┤
│  CAPA 4: MOTORA (Salida)                                    │
│  GitGuard → FrontmatterWriter → LinkInjector                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Stack Tecnológico Final

-   **Core:** Python 3.11+ (Termux/Ubuntu).
-   **Seguridad:** Git (GitPython) + SecurityGuard (PII/Prompt-Injection filters).
-   **Ingesta:** `watchdog`, `python-frontmatter`.
-   **Memoria:** SQLite (Indice + Embeddings serializados).
-   **Cognitiva:** Ollama (`qwen2.5:3b` para texto, `nomic-embed-text` para vectores).
-   **CLI:** `typer` + `rich`.
-   **Notificaciones:** `termux-api` (automático en Termux).

---

## 4. Estructura del Proyecto (Final)

```text
grimoire/
├── ingest/           # observer.py, parser.py
├── cognition/        # llm_router.py, tagger.py, embedder.py, connector.py, oracle.py
├── memory/           # db.py, taxonomy.py
├── output/           # git_guard.py, frontmatter_writer.py, link_injector.py
└── utils/            # config.py, logger.py, hashing.py, notifications.py, security.py, backup.py, system.py
```

---

## 6. Roadmap de Desarrollo (Estado de Ejecución)

### **Fase 0: Safety Harness** ✅
- [x] Inicializar repo y estructura.
- [x] Implementar `git_guard.py` (autocommit).
- [x] Soporte `--dry-run` global.
- [x] Logger estructurado.

### **Fase 1: Ingesta Idempotente** ✅
- [x] `watchdog` con debounce real.
- [x] Parser frontmatter/cuerpo.
- [x] `content_hash` idempotente.
- [x] Schema SQLite funcional.

### **Fase 2: Cognición Local** ✅
- [x] Integración Ollama (`llm_router.py`).
- [x] Tagger con schema JSON estricto.
- [x] `taxonomy.py` (normalización de tags).

### **Fase 3: Memoria Vectorial + Conexiones** ✅
- [x] Embedder (Ollama + Cosine similarity en Python).
- [x] Almacenamiento vectorial en SQLite.
- [x] `link_injector.py` (Conexiones Sugeridas).

### **Fase 4: El Oráculo (RAG CLI)** ✅
- [x] Comando `grimoire ask` (RAG).
- [x] Citas automáticas `[[nota]]`.
- [x] Exportación a nuevas notas.

### **Fase 5: Daemon + Notificaciones** ✅
- [x] `grimoire daemon start/stop/status` (background).
- [x] Notificaciones `termux-api`.
- [x] Dashboard de métricas en `grimoire status`.

### **Fase 6: Hardening & Seguridad** ✅
- [x] Políticas de privacidad por nota (`privacy: local`).
- [x] Filtro de contenido sensible (PII/Keys).
- [x] Sanitización de prompts.
- [x] Backup automático de SQLite (rotación diaria).

---

## 7. Próximos Pasos (Expansiones Futuras)

1.  **7.1 El Espejo Negro:** Detector de contradicciones filosóficas.
2.  **7.2 MCP Server:** Exponer el Grimorio a Claude Desktop/Web.
3.  **7.3 Ingesta Multi-Formato:** Soporte para PDF/EPUB.
4.  **7.7 Modo Alquimista:** Síntesis creativa de múltiples notas.

---

## 8. Decisiones Tomadas

1.  **Entorno:** Dual (Termux + Ubuntu).
2.  **LLM:** `qwen2.5:3b`.
3.  **Privacidad:** Manual (Opt-in via frontmatter).
4.  **Bóveda:** Única.
5.  **Embeddings:** 100% Python/Ollama (sin dependencias de torch/numpy para máxima portabilidad).

---

*Project Grimoire v2.0 is alive. Raw cognition, total privacy.*
