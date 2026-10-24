# Grimore — Guía de Usuario (Español)

> Un motor de conocimiento local-first para bóvedas de documentos —
> Markdown, PDF, EPUB, DOCX, ODT, RTF, HTML, TXT — todos procesados por
> el mismo pipeline.
> Todo se ejecuta contra una instancia de Ollama en loopback; sin claves de
> API, sin telemetría, sin nube.

---

## Índice

1. [Qué es Grimore (y qué no es)](#1-qué-es-grimore-y-qué-no-es)
2. [Requisitos](#2-requisitos)
3. [Instalación](#3-instalación)
4. [Tu primera bóveda — recorrido de cinco minutos](#4-tu-primera-bóveda--recorrido-de-cinco-minutos)
5. [El archivo `grimore.toml`](#5-el-archivo-grimoretoml)
   - [Formatos soportados](#formatos-soportados)
   - [Sidecars: dónde se guardan los metadatos no-MD](#sidecars-dónde-se-guardan-los-metadatos-no-md)
   - [Motores opcionales: backends de PDF, OCR, sniffer de magic-bytes](#motores-opcionales-backends-de-pdf-ocr-sniffer-de-magic-bytes)
   - [Perfiles multi-bóveda](#perfiles-multi-bóveda)
6. [Comandos del día a día](#6-comandos-del-día-a-día)
   - [`scan`](#scan)
   - [`connect`](#connect)
   - [`ask`](#ask)
   - [`eval`](#eval)
   - [`tags`](#tags)
   - [`dedupe`](#dedupe)
   - [`prune`](#prune)
   - [`status`](#status)
   - [`preflight`](#preflight)
   - [`pilot`](#pilot)
   - [`daemon`](#daemon)
   - [`maintenance run`](#maintenance-run)
   - [`migrate-embeddings`](#migrate-embeddings)
   - [`category`](#category)
   - [`chronicler`](#chronicler)
   - [`mirror`](#mirror)
   - [`distill`](#distill)
   - [`graph export`](#graph-export)
   - [`mcp`](#mcp)
   - [`serve`](#serve)
7. [La shell interactiva — `grimore shell`](#7-la-shell-interactiva--grimore-shell)
   - [Composición de la entrada](#composición-de-la-entrada)
   - [Comandos slash](#comandos-slash)
   - [Menciones con `@`](#menciones-con-)
   - [Notas fijadas (pins)](#notas-fijadas-pins)
   - [Confirmaciones de seguridad](#confirmaciones-de-seguridad)
   - [Guardar transcripciones](#guardar-transcripciones)
   - [Persistencia de conversaciones — `/thread`, `/resume`, `/threads`](#persistencia-de-conversaciones--thread-resume-threads)
   - [Memoria conversacional y `/forget`](#memoria-conversacional-y-forget)
   - [Cambiar de modelo en vivo](#cambiar-de-modelo-en-vivo)
   - [Barra inferior](#barra-inferior)
   - [Modo vi](#modo-vi)
8. [Taxonomía: `taxonomy.yml`](#8-taxonomía-taxonomyyml)
9. [Convenciones de frontmatter](#9-convenciones-de-frontmatter)
10. [Privacidad y seguridad](#10-privacidad-y-seguridad)
11. [Integraciones — MCP, API HTTP, backends compatibles con OpenAI](#11-integraciones--mcp-api-http-backends-compatibles-con-openai)
12. [Trabajar con modelos grandes](#12-trabajar-con-modelos-grandes)
13. [Solución de problemas](#13-solución-de-problemas)
14. [Glosario](#14-glosario)

---

## 1. Qué es Grimore (y qué no es)

Grimore observa un directorio de documentos (tu **bóveda**) y
lo convierte en una base de conocimiento consultable. Markdown es el
ciudadano de primera clase, pero **PDF, EPUB, DOCX, ODT, RTF, HTML y
TXT** se extraen con el mismo pipeline. Para cada documento:

- extrae **etiquetas** y un **resumen** de un párrafo con un LLM local,
- la archiva bajo una **categoría** dentro de un árbol jerárquico,
- divide el cuerpo en **fragmentos** y embebe cada uno como un **vector**,
- propone **wikilinks** entre notas semánticamente relacionadas,
- responde preguntas en lenguaje natural con **citas a tus propias notas**.

Todo se sostiene sobre una base de datos **SQLite** con modo WAL y un
índice **FTS5**. La recuperación fusiona **BM25** (texto completo) y
**similitud coseno** (vectores) mediante **Reciprocal Rank Fusion** (RRF).

**Lo que Grimore no es:**

- No es un editor de notas. Usa Obsidian / Logseq / tu editor preferido.
- No es un servicio de sincronización en la nube. Combínalo con `git`,
  Syncthing, o lo que ya uses.
- No es un "agente de IA" que edite tus notas de forma autónoma. Toda
  operación destructiva opera por defecto en modo **dry-run**.

---

## 2. Requisitos

| Componente | Mínimo | Notas |
| :--- | :--- | :--- |
| Python | 3.11 | Se usan anotaciones de tipo modernas. |
| Ollama | Última versión estable | Escuchando en `127.0.0.1:11434`. |
| Disco | ~1.5 GB por modelo embebedor pequeño | Los modelos viven bajo `~/.ollama`. |
| RAM | 8 GB | Para `qwen2.5:3b` + `nomic-embed-text`. Modelos de 14B piden 16 GB+. |
| `git` | Cualquier versión moderna | Necesario para la red de seguridad. |

Sistemas soportados: **Linux** y **Windows** son los objetivos principales.
**Termux en Android** es un entorno alternativo soportado.

Modelos de chat recomendados (locales):

- `qwen2.5:3b` — el predeterminado documentado; rápido y bien portado en CPU.
- `qwen3.5:0.8b` — aún más rápido, pero emite una fase de "thinking" que el
  Oráculo debe esperar antes de que aparezcan tokens.
- `ministral-3:14b` — más lento pero con mejor razonamiento. Sube
  `cognition.request_timeout_s` a 600+ antes de usarlo.

Modelo embebedor recomendado:

- `nomic-embed-text` — el predeterminado documentado.
- `embeddinggemma:latest` — alternativa si prefieres el codificador de Google.

---

## 3. Instalación

```bash
git clone https://github.com/kahz12/Grimore-MD.git
cd Grimore-MD
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

Confirma que todo está conectado:

```bash
grimore preflight
```

`preflight` verifica cuatro cosas, en este orden:

1. El archivo de configuración carga sin errores.
2. Ollama es accesible.
3. Los modelos de chat y embeddings configurados están realmente descargados.
4. La ruta de la bóveda existe y (salvo con `--no-check-git`) es un repo git.

Cualquier error bloqueante termina con código 1 y un panel que te dice
exactamente qué línea revisar.

---

## 4. Tu primera bóveda — recorrido de cinco minutos

Supongamos que tus notas viven en `~/notas`.

```bash
cd ~/notas
git init                                            # red de seguridad
echo "vault.path = \"~/notas\"" > grimore.toml      # ver §5 para el archivo completo

grimore preflight                                   # panel verde = listo
grimore scan --dry-run                              # previsualizar el etiquetado
grimore scan --no-dry-run                           # escribir etiquetas + índice
grimore connect --no-dry-run                        # añadir wikilinks
grimore ask "qué temas atraviesan mis notas sobre memoria?"
```

Ya tienes una bóveda etiquetada, embebida, con enlaces inyectados y has
formulado tu primera pregunta al Oráculo. Desde aquí puedes seguir con los
comandos puntuales o abrir la **shell interactiva** (§7) y quedarte dentro
de ella:

```bash
grimore shell
```

---

## 5. El archivo `grimore.toml`

Grimore busca `grimore.toml` en el directorio actual de trabajo. Los
valores por defecto son seguros: nada sale de tu máquina y el modo de
escritura es opt-in. Un ejemplo anotado completo:

```toml
[vault]
path = "./vault"
ignored_dirs = [".obsidian", ".trash", ".git", "Templates"]
# Formatos de documento que Grimore recogerá, por extensión en
# minúsculas. El valor por defecto incluye todos los adaptadores
# pre-Python sin dependencias del sistema operativo. Añade "doc" si
# tienes antiword instalado.
formats = ["md", "txt", "html", "htm", "docx", "pdf", "epub", "rtf", "odt"]
# Carpeta oculta, relativa a la bóveda, donde Grimore guarda un sidecar
# `.md` por cada documento que no sea Markdown. Los binarios originales
# nunca se modifican; las etiquetas, resúmenes y bloques de enlaces
# sugeridos viven aquí.
sidecar_dir   = ".grimore/sidecars"
# Si es false, los metadatos no-Markdown permanecen solo en la base de
# datos (no se escriben sidecars en disco).
write_sidecars = true
# Etiqueta opcional mostrada en la barra inferior de la shell. Si no se
# define, se usa el nombre del directorio.
display_name = "Biblioteca"

[cognition]
model_llm_local        = "qwen2.5:3b"
model_embeddings_local = "nomic-embed-text"
allow_remote           = false  # bloquea endpoints Ollama / OpenAI que no sean loopback
hybrid_search          = true   # BM25 + coseno vía RRF
rrf_k                  = 60     # curva de peso por rango; menor = más pronunciada
bm25_weight            = 0.25   # peso de la lista por palabras en la fusión; 1 = RRF clásico
connect_threshold      = 0.7    # umbral coseno para sugerir un wikilink
dedupe_threshold       = 0.90   # umbral coseno para un par casi-duplicado
request_timeout_s      = 300    # /api/generate, vía JSON
stream_timeout_s       = 300    # /api/generate, vía streaming
embed_timeout_s        = 30     # /api/embeddings
# Comprobación experimental de suficiencia antes de redactar: una llamada
# extra al modelo. No garantiza respaldo semántico. Ante fallo o evidencia
# insuficiente no genera hechos; puede rechazar preguntas respondibles.
# Con la opción activa, cada bloque de respuesta requiere referencias explícitas;
# la transmisión espera la validación completa.
answer_sufficiency_check = false
answer_sufficiency_timeout_s = 180  # entero positivo, por comprobación
# Segunda pasada de re-ranking tras la fusión RRF. Desactivada por
# defecto — añade trabajo por consulta. "llm" pide al modelo de chat
# que puntúe cada candidato (sin instalación extra, ~15–30 s/consulta);
# "cross-encoder" usa un reranker sentence-transformers (sub-segundo;
# necesita el extra `reranker`).
rerank          = false
rerank_pool     = 20
rerank_engine   = "llm"          # "llm" | "cross-encoder"
rerank_model    = "BAAI/bge-reranker-base"
# Backend vectorial. "auto" (defecto) usa sqlite-vec cuando la
# extensión está cargada y la dimensión coincide; cae a numpy
# matmul en otro caso. Forzar "numpy" fija la vía en memoria;
# "sqlite-vec" exige el extra opcional y que la wheel cargue.
vector_backend  = "auto"
# Backend de LLM. "ollama" (defecto) conserva el comportamiento v2.x.
# Establece "openai" para hablar con cualquier servidor compatible con
# OpenAI (llama.cpp, vLLM, LM Studio, OpenRouter, OpenAI). Ver §11
# para los detalles completos.
llm_backend     = "ollama"
llm_base_url    = ""              # p. ej. "http://localhost:8080/v1" para llama.cpp server
llm_api_key_env = "GRIMORE_LLM_API_KEY"   # variable de entorno con el token bearer
# Ajustes finos. Todos son opcionales: omitirlos reproduce exactamente el
# comportamiento histórico, que es el valor mostrado aquí.
chunk_store_chars = 500      # texto del chunk espejado en la tabla embeddings.
                             # Es lo que ve el re-ranker cross-encoder, así que
                             # subirlo mejora su precisión a costa de tamaño de
                             # base y de memoria en cada carga completa.
context_max_chars = 16000    # tope del contexto unido que se inyecta al LLM.
                             # Los modelos con ventana grande admiten más.
embed_batch_size  = 32       # textos por petición a /api/embed. Lotes mayores
                             # ahorran round-trips en servidores con batching.
circuit_failure_threshold = 5    # fallos seguidos antes de abrir el circuito
circuit_cooldown_s        = 120  # segundos de corte antes de reintentar
conditional_rewrite = true   # reescribe una pregunta de seguimiento en una
                             # consulta autónoma SÓLO cuando apunta al turno
                             # anterior (pronombres, demostrativos, una
                             # conjunción inicial, o menos de 5 palabras). Una
                             # pregunta que nombra su propio tema recupera lo
                             # mismo con o sin reescritura, así que se ahorra
                             # el round-trip. Ponla en false para reescribir
                             # siempre que haya historial.
rewrite_timeout_s = 60       # presupuesto sólo para esa reescritura. Corre
                             # antes de recuperar, con la pantalla en blanco,
                             # así que no debe heredar request_timeout_s. Al
                             # expirar se usa la pregunta original. Sus fallos
                             # no cuentan para el circuit breaker: son trabajo
                             # opcional y abrirlo cancelaría también la
                             # generación de respuestas.
vec_matrix_cache  = true     # guarda la matriz de puntuación densa como .npy
                             # junto a la base de datos y la recarga mapeada en
                             # memoria. Sin ella, cada `grimore ask` de un solo
                             # uso la reconstruye leyendo todos los vectores.
                             # Ocupa 4 bytes por dimensión y chunk (~53 MB para
                             # 17.500 chunks de 768 dimensiones). Ponla en false
                             # para volver al comportamiento anterior; al
                             # hacerlo se borra el fichero existente.

[ingest]
# Motor de PDF. "pypdf" es el predeterminado siempre disponible;
# "pdfplumber" y "pymupdf" son extras opcionales que preflight reporta.
# PyMuPDF es AGPL — verifica la compatibilidad de licencia antes de
# usarlo.
pdf_engine    = "pypdf"
# OCR como respaldo para PDFs escaneados (páginas sin capa de texto).
# Desactivado por defecto; requiere el extra `ocr`, `tesseract` y el
# Poppler (`pdfinfo` y `pdftoppm` o `pdftocairo`) en el PATH.
ocr           = false
ocr_language  = "eng" # "spa" para español; "spa+eng" para combinar idiomas
# Segundos por etapa y página: rasterización y luego reconocimiento.
# Debe ser un entero positivo; una página puede consumir ambos límites.
ocr_timeout_s = 30
# Intentos máximos de OCR por PDF; 0 mantiene el procesamiento sin tope.
# Los fallos cuentan; las páginas digitales legibles no consumen el límite.
ocr_max_pages = 0
# Sniffer de magic-bytes para archivos sin extensión o mal nombrados.
# Desactivado por defecto; requiere el extra `sniff` (python-magic) y
# libmagic del sistema.
sniff_magic   = false
# Motor de chunking. "markdown" (defecto) divide por encabezados y
# tope de tamaño, determinista y sin coste durante el scan.
# "semantic" embebe cada oración y corta en cambios de tema —
# mejor recall en prosa larga, a costa de un embedding por oración
# durante el scan.
chunker            = "markdown"
semantic_threshold = 0.55     # caída de coseno que abre un nuevo chunk
chunk_max_chars    = 1500     # tope estricto sea cual sea el motor

[memory]
db_path = "grimore.db"

[output]
auto_commit = true  # commit git antes de cada escritura
dry_run     = true  # valor seguro — actívalo con --no-dry-run

[maintenance]
enabled         = true
interval_hours  = 24
vacuum          = true
purge_tags      = true
wal_checkpoint  = true

[chronicler.windows]
"tech/"          = 90
"tools/"         = 90
"infra/"         = 90
"dev/"           = 180
"code-snippets/" = 180
"concepts/"      = 0    # 0 = nunca obsoleto
"theory/"        = 0
"journal/"       = 0
"daily/"         = 0

[daemon]
enabled         = false
log_events      = true
debounce_seconds = 45    # agrupa ráfagas de guardado del editor en un re-index
poll_fallback    = false # usa polling en vez de inotify (NFS, FUSE, Termux)
poll_interval_s  = 30.0

[shell]
vi_mode         = false  # true para activar modo vi en prompt_toolkit
fuzzy_threshold = 55     # 0–100, puntuación rapidfuzz mínima para autocompletado @
# Dónde guarda `/thread save` las transcripciones de conversación.
# Las rutas relativas se anclan bajo tu home, así que el valor por
# defecto vive en ~/.grimore/threads. Las rutas absolutas se usan
# tal cual.
threads_dir     = ".grimore/threads"
# Turnos previos de Q&A que se conservan como contexto conversacional para la
# reescritura de consulta del Oracle. Cada turno se antepone al prompt, así que
# una ventana grande cuesta tokens en cada follow-up. 0 desactiva la memoria.
max_turns       = 3

# Opcional: perfiles de bóveda nombrados. Cada bloque
# [profiles.<nombre>] se fusiona profundamente con las secciones de
# nivel superior cuando se activa con `--profile <nombre>` o
# `GRIMORE_PROFILE=<nombre>`. Ver §5 → "Perfiles multi-bóveda".
# [profiles.trabajo]
# [profiles.trabajo.vault]
# path = "/home/yo/notas-trabajo"
# [profiles.trabajo.cognition]
# model_llm_local = "qwen2.5:14b"
```

Las claves desconocidas se registran como WARNING y se ignoran — copiar
fragmentos de un README antiguo no romperá el CLI.

### Formatos soportados

| Extensión | Adaptador          | Notas                                                                       |
|-----------|--------------------|-----------------------------------------------------------------------------|
| `md`      | Markdown           | Primera clase. Frontmatter, wikilinks y escritura inline.                   |
| `txt`     | Texto plano        | Solo stdlib. La primera línea no vacía se usa como título.                  |
| `html`    | HTML / XHTML       | `<title>`, luego el primer `<h1>` / `<h2>`. Requiere `beautifulsoup4`.      |
| `htm`     | HTML (alias)       | Mismo adaptador que `.html`.                                                |
| `docx`    | DOCX (OOXML)       | Pure-stdlib: `zipfile` + `xml.etree`. Los encabezados generan secciones.    |
| `pdf`     | PDF                | Secciones por página; el ancla de página llega a las citas como `#p.N`.     |
| `epub`    | EPUB               | Capítulos en orden de spine → secciones; títulos desde metadatos OPF.       |
| `rtf`     | RTF                | `striprtf`; una sola sección sin ancla.                                     |
| `odt`     | OpenDocument Text  | Pure-stdlib zip+XML. Sigue el mismo flujo que DOCX para esquemas ODT.       |
| `doc`     | `.doc` heredado    | **Opt-in.** Añade `"doc"` a `formats` e instala `antiword` en el PATH.      |

Para formatos paginados, las citas obtienen un ancla automáticamente —
una respuesta extraída de la página 137 de *Designing Data-Intensive
Applications* se renderiza como
`[[Designing Data-Intensive Applications#p.137]]`. Para formatos
estructurados, el ancla cae al encabezado más cercano.

### Sidecars: dónde se guardan los metadatos no-MD

Grimore nunca modifica un PDF / EPUB / DOCX / ODT / RTF / HTML / TXT
original. En su lugar, la capa de cognición escribe un **sidecar** `.md`
bajo `<bóveda>/<sidecar_dir>/` (por defecto: `.grimore/sidecars/`) que
refleja la ruta del documento:

```
vault/
  Libros/Designing-Data-Intensive-Applications.pdf
  .grimore/
    sidecars/
      Libros/Designing-Data-Intensive-Applications.pdf.md
```

El sidecar contiene el frontmatter (`tags`, `summary`, `category`,
`last_tagged`) y — una vez ejecutado `grimore connect` — un bloque
`## Suggested Connections`. La extensión `.pdf` original se conserva en
el nombre del sidecar para que `Foo.pdf` y `Foo.epub` nunca colisionen
en un único `Foo.md`. Configura `write_sidecars = false` para mantener
todo solo en la base de datos.

Las menciones `@` en la shell son conscientes de los sidecars: escribir
`@mi-libro` resuelve al binario original, pero el adjunto que Grimore
le da al modelo proviene del texto extraído del sidecar.

### Motores opcionales: backends de PDF, OCR, sniffer de magic-bytes

Estas opciones viven bajo `[ingest]` y permanecen apagadas hasta que
las actives. Preflight (`grimore preflight`) solo sondea las que has
habilitado, así que una instalación por defecto reporta una fila ✓
limpia por formato.

- **Motores de PDF alternativos.** `pdfplumber` añade lectura por bloques
  y filas de tablas delimitadas, y recupera explícitamente el texto anterior
  ante diseños que no soporta. `pymupdf` es otro motor opcional (AGPL —
  verifica compatibilidad de licencia). Instala el extra
  correspondiente:

  ```bash
  pip install 'grimore[pdf-plumber]'
  pip install 'grimore[pdf-mupdf]'   # AGPL-3.0
  ```

  Luego configura `pdf_engine` como `"pdfplumber"` o `"pymupdf"`. El
  cambio se aplica en el siguiente scan sin reiniciar.
  La revisión de lectura de `pdfplumber` también fuerza una nueva extracción
  al volver a escanear los PDF previamente indexados, sin modificar el original.

- **OCR para PDFs escaneados.** Las páginas con una capa de texto vacía
  pueden rasterizarse y procesarse con OCR vía `tesseract`. Se
  necesitan el extra de Python (`pip install 'grimore[ocr]'`), el binario
  `tesseract` y Poppler (`pdfinfo` y `pdftoppm` o `pdftocairo`). Luego activa
  `ocr = true`. Las secciones generadas por
  OCR llevan el encabezado sintético `(ocr)` para que los revisores
  puedan auditarlas. Selecciona el idioma con `[ingest].ocr_language = "spa"`
  o `"spa+eng"` para documentos mixtos; se conserva `"eng"` por defecto.
  Todos los datos de idioma solicitados deben estar instalados (en Linux,
  `sudo apt-get install tesseract-ocr-spa` añade español). La comprobación
  previa avisa si faltan; no se sustituye el idioma silenciosamente. Cambiar
  el idioma con OCR activo vuelve a extraer PDFs sin modificar sus bytes en
  el siguiente escaneo, conservando los vectores de pasajes idénticos.

**Cobertura de extracción de PDF.** Grimore registra por página si falló el
motor, si no había texto extraíble o si se usó OCR. El diagnóstico
`extraction` del sidecar y de `grimore_get_note` indica `complete` (sin
incidencias), `recovered` (texto recuperado, por ejemplo mediante OCR) o
`partial` (queda al menos una página sin texto). La vista de fuente web
advierte sobre páginas afectadas; un documento que ya no puede leerse se
muestra como `failed`. Una página deliberadamente vacía también aparece
como `partial`: conviene revisarla antes de asumir que falta contenido.
Si se solicitó OCR y no se recuperó texto, el diagnóstico puede añadir una
causa: herramientas no disponibles, renderizado fallido, reconocimiento fallido
o ausencia de texto reconocido. La vista web muestra esa causa sin exponer
mensajes internos del motor.
Un PDF sin texto recuperable conserva su registro y su sidecar con estado
`partial`, pero no genera fragmentos de búsqueda ni pide al modelo que invente
etiquetas o resumen a partir de un cuerpo vacío.
Si faltan herramientas OCR o falla el renderizado o reconocimiento, el siguiente escaneo reintenta
ese PDF sin exigir que cambie el archivo. Cuando el texto sigue igual, no
repite la clasificación ni los vectores idénticos.
La vista de fuente y MCP describen la lectura actual del archivo; el
sidecar conserva el diagnóstico del último procesamiento efectivo. Si cambian
los bytes de un PDF pero el texto extraído sigue idéntico, el siguiente
escaneo refresca el diagnóstico y la huella del sidecar existente sin repetir
el etiquetado; también actualiza las anclas de página del índice. Reutiliza
los vectores de fragmentos idénticos y solo calcula los que cambian al variar
los límites de página. Preserva las notas manuales del sidecar. Ninguno de estos
diagnósticos garantiza por sí solo que el índice tenga todos los pasajes:
consulta el estado de la fuente antes de usarlos como evidencia.

- **Sniffer de magic-bytes.** Cuando `sniff_magic = true`, los archivos
  cuya extensión falta o no coincide con `formats` se inspeccionan por
  su tipo de contenido real (vía `libmagic`) y se enrutan al adaptador
  correcto si existe. Útil para bóvedas con descargas ad-hoc. Instala
  el extra `sniff` y `libmagic` del sistema (Linux:
  `apt install libmagic1`; Termux: `pkg install file`; Windows:
  `pip install python-magic-bin`).

Para comparar motores antes de activarlos en una bóveda, ejecuta
`venv/bin/python eval/benchmark_extraction.py --repeat 3 --include-legacy`
desde el repositorio. Repite la medición en cada equipo: el tiempo y la
memoria dependen del hardware.

### Perfiles multi-bóveda

Un único `grimore.toml` puede contener varios **perfiles** nombrados.
Cada bloque `[profiles.<nombre>]` aloja overrides que *se fusionan
profundamente* sobre las secciones de nivel superior — lo que no
sobreescribas se hereda. Elige un perfil por invocación con
`--profile <nombre>` (en cualquier comando), o establece la variable
de entorno `GRIMORE_PROFILE` para fijar el predeterminado.

```toml
# Las claves de nivel superior siguen aplicando cuando no se selecciona perfil.
[vault]
path = "./vault"
display_name = "Personal"

[cognition]
model_llm_local = "qwen2.5:3b"

# Una máquina de oficina más potente + una bóveda distinta.
[profiles.trabajo]
[profiles.trabajo.vault]
path = "/home/yo/notas-trabajo"
display_name = "Trabajo"
[profiles.trabajo.cognition]
model_llm_local = "qwen2.5:14b"
request_timeout_s = 600
```

```bash
grimore --profile trabajo ask "¿qué cambió en los standups de la semana pasada?"
GRIMORE_PROFILE=trabajo grimore shell
```

Precedencia: el flag `--profile`/`-P` gana sobre `GRIMORE_PROFILE`,
que a su vez gana sobre los valores por defecto sin perfil. Un
perfil desconocido falla con un mensaje claro en lugar de volver
silenciosamente a los valores por defecto. `grimore status` muestra
el nombre del perfil activo junto a la ruta de la bóveda.

Las listas se reemplazan por completo, no se concatenan —
`formats = ["md"]` dentro de un perfil significa *solo* Markdown,
no "Markdown también".

---

## 6. Comandos del día a día

Ejecuta `grimore <cmd> --help` para la lista autoritativa de flags. A
continuación, un recorrido detallado.

### `scan`

```bash
grimore scan [-p RUTA] [--dry-run|--no-dry-run] [--json]
```

Recorre la bóveda, etiqueta documentos nuevos o modificados en cada
formato configurado y refresca el índice de embeddings.

- `-p, --vault-path` — anula la ruta de la bóveda solo para esta ejecución.
- `--dry-run` / `--no-dry-run` — sobrescribe la config sin tocarla.
- `--json` — emite logs estructurados en JSON (útil en CI / monitoreo).

**Idempotencia.** Hashing en dos niveles: un SHA-256 barato sobre los
bytes del archivo (`file_hash`) actúa como puerta del paso costoso de
extracción; normalmente solo los documentos cuyos bytes han cambiado pagan
la re-extracción. Dentro de eso, un `content_hash` sin cambios omite la
llamada al LLM por completo. La excepción son los PDF: el índice conserva
una huella del motor, OCR y su tiempo límite activo. Al cambiar cualquiera
de esos ajustes, el siguiente `scan` vuelve a extraer los PDF afectados
aunque sus bytes no cambien, y actualiza sus pasajes y anclas de página.
Los vectores de fragmentos cuyo texto sigue idéntico se reutilizan.
Reescanear una bóveda sin cambios de archivos ni de extracción sigue
siendo prácticamente gratis incluso para una biblioteca de PDFs de 500 MB.
`--dry-run` muestra el reprocesamiento pendiente sin actualizar la huella.

**Qué se escribe.** Con `--no-dry-run`:

- `tags`, `summary`, `category`, `last_tagged` se escriben en el
  frontmatter YAML de cada nota,
- en documentos de más de 4.000 caracteres, `section_analysis` conserva las
  páginas o encabezados de los tramos consultados, sus etiquetas y resúmenes,
  las fuentes utilizadas para el resumen global y los recuentos de cobertura;
  se consultan hasta 12 tramos repartidos por el documento y el resto se
  declara como `unanalyzed_parts` (no se presenta como análisis completo),
- la fila SQLite se actualiza (upsert),
- los embeddings por fragmento se almacenan con clave
  `sha256(modelo ‖ fragmento)` (cambiar el modelo embebedor invalida la
  caché limpiamente).

**Salida de privacidad.** Una nota con `privacy: never_process` en su
frontmatter se omite antes de invocar al LLM.

### `connect`

```bash
grimore connect [--dry-run|--no-dry-run] [-t UMBRAL]
```

Recorre cada nota y encuentra hermanas semánticamente similares por
coseno. Con `--no-dry-run`, mantiene de forma idempotente un bloque
`## Suggested Connections` al final de cada nota con wikilinks a sus
mejores coincidencias.

- `-t, --threshold` — piso coseno en `[0.0, 1.0]`. El predeterminado
  proviene de `cognition.connect_threshold` (0.7). Menor = más
  sugerencias (más ruidosas).

El bloque se regenera, no se concatena — ejecutar `connect` repetidamente
es seguro.

### `ask`

```bash
grimore ask "<pregunta>" [-k N] [-e RUTA]
```

Una consulta puntual de generación aumentada por recuperación (RAG).

- `-k, --top-k` — cuántos fragmentos de contexto recuperar (por defecto 5).
  Más fragmentos = respuesta más rica, pero modelo más lento.
- `-e, --export RUTA` — renderiza la respuesta + fuentes como una nota
  Markdown en `RUTA` en lugar de transmitir a stdout.

El Oráculo siempre cita los documentos de los que extrajo contexto. Las
citas aparecen al final de la respuesta como enlaces `[[título]]` — y
para formatos paginados o estructurados obtienen un ancla:
`[[Designing Data-Intensive Applications#p.137]]` para PDFs,
`[[Informe Anual#Capítulo 3]]` para DOCX, EPUB y ODT. Las citas de
Markdown y TXT permanecen sin ancla.

Las citas alucinadas (títulos que el modelo se inventa y que no
estaban en el contexto recuperado) se eliminan del cuerpo
renderizado y quedan registradas en `oracle_citation_hallucinated`.
La lista de fuentes devuelta refleja siempre las notas *recuperadas*,
no lo que afirmara el modelo.

Acota qué se busca con `--category`, `--tag` y `--format`:

```bash
grimore ask "¿política de retención?" --category infra
grimore ask "¿política de retención?" --tag compliance --tag security   # ambos
grimore ask "¿qué dicen los PDF?" --format pdf
```

Los filtros restringen qué notas se buscan siquiera, así que el Oracle sólo
puede citar lo que sobreviva. Se combinan con AND, y los `--tag` repetidos
también. `--category` incluye descendientes, de modo que `infra` cubre además
`infra/networking`. Si no casa nada, el comando lo dice y para, en vez de
responder en silencio desde todo el vault.

No hay filtro por fecha. Los únicos sellos de tiempo de una nota registran
cuándo la tocó el escáner, no nada del documento: tras un scan todas comparten
sello, así que un filtro "modificado después de" ordenaría por orden de escaneo.

### `eval`

```bash
grimore eval [-g RUTA] [-k N] [--retrieval-k N] [--retrieval-only]
             [--baseline] [--judge/--no-judge] [--export RUTA]
             [--history RUTA] [--compare RUTA] [--checkpoint RUTA] [--json]
```

Corre un conjunto Q&A dorado contra el Oráculo y reporta métricas de
recuperación y calidad de respuesta. Por defecto usa
`eval/grimore_golden.yaml`.

| Métrica | Qué mide |
|---|---|
| **hit@1 / hit@3** | Si una fuente esperada es el primer resultado / está entre los 3 primeros. |
| **recall@k** | Fracción de fuentes esperadas que aparecen en el top-k recuperado. |
| **MRR** | Rango recíproco medio de la primera fuente esperada acertada. |
| **validez de citas** | Fracción de citas emitidas que apuntan a una fuente recuperada. Vacía (no 1,0) si la respuesta no cita nada; `faithfulness` es un alias obsoleto. |
| **respuestas citadas** | Fracción de preguntas con respuesta cuya respuesta cita al menos una fuente. |
| **evidencia en contexto / citada / seleccionada** | Si el pasaje esperado llegó al modelo, se citó con su página o encabezado y fue el fragmento exacto que eligió la cita. |
| **abstención** | Indicios de rechazo en preguntas sin respuesta y en las que sí la tienen (rechazos indebidos). Son heurísticos; el informe conserva una cola de revisión humana. |
| **keyword recall** | Fracción de palabras clave esperadas presentes en la respuesta. |
| **answer relevance** | LLM como juez: el modelo local puntúa 0–10 la respuesta vs la pregunta. |
| **latencia p50 / p95** | Tiempo de reloj por turno. |

- `-g, --golden` — ruta al YAML del conjunto dorado.
- `-k, --top-k` — pasajes que alimentan la respuesta por pregunta (defecto 5).
- `--retrieval-k` — profundidad del pool rankeado que puntúa Hit@k / MRR,
  independiente de `--top-k` (defecto 10).
- `--retrieval-only` — puntúa solo el ranking y salta la generación de
  respuestas. Rápido, determinista y sin modelo de chat — córrelo en
  cada cambio.
- `--baseline` — corre dos veces, híbrido (RRF) vs solo-denso, y muestra
  el delta por métrica para ver qué aporta la fusión con BM25.
- `--no-judge` — salta el paso del LLM-juez (útil sin red / CI rápido).
- `--export` — vuelca el informe completo como JSON.
- `--history` — añade un registro de una línea a un ledger JSONL, para
  seguir la tendencia de las métricas entre corridas.
- `--compare` — compara esta corrida contra un JSON previo de `--export`
  y sale con código distinto de cero ante cualquier regresión. Combinado
  con `--retrieval-only` es una puerta de calidad barata para CI:

  ```bash
  grimore eval --retrieval-only --export run.json --compare last-good.json
  ```

- `--checkpoint` — guarda cada caso terminado en un archivo JSONL y retoma
  desde él al repetir el mismo comando. Úsalo en ejecuciones largas con un
  modelo real: una interrupción cuesta como mucho el caso en curso. Los
  casos en que el modelo no respondió se repiten, y un archivo de otra
  ejecución (preguntas, configuración, corpus, código u opciones) se
  rechaza. Un informe con `generation_failure_n` mayor que cero contó
  fallos del modelo como rechazos; repite esos casos antes de usarlo.
- `--json` — logs estructurados en JSON.

El formato del conjunto dorado es una entrada por Q&A con
`follow_ups` opcionales para ejercitar también la memoria
conversacional (`Oracle.history`). La versión de schema está
fijada; las claves desconocidas elevan error para que las
erratas afloren al instante. Consulta `eval/grimore_golden.yaml`
para un ejemplo funcional.

### `tags`

```bash
grimore tags [-n N]
```

Tabla de frecuencia de todas las etiquetas en uso. `-n` limita las filas
mostradas (por defecto 30).

### `dedupe`

```bash
grimore dedupe [-t UMBRAL] [-n N] [-e reporte.json]
```

Encuentra notas duplicadas en dos niveles: copias **exactas** (mismo hash
de contenido — re-importaciones, backups sueltos) y **casi-duplicados**
(pares de notas cuyos vectores promediados de chunks superan un umbral de
similitud coseno — reescrituras, ediciones tipo `archivo (1).md`). Solo
reporta: no toca ni el vault ni el índice. Determinista y sin LLM, así que
funciona aunque Ollama esté ocupado.

- `-t, --threshold` — coseno mínimo para reportar un par, en `[0.0, 1.0]`
  (por defecto `[cognition].dedupe_threshold`, 0.90).
- `-n, --limit` — máximo de pares casi-duplicados mostrados, mejores
  primero (por defecto 30).
- `-e, --export` — escribe el reporte completo (ambas secciones) en JSON.

### `prune`

```bash
grimore prune [-p RUTA] [--dry-run|--no-dry-run]
```

Elimina filas DB para notas que ya no existen en disco y luego depura
filas de etiquetas sin referencias. **Dry-run por defecto** — pasa
`--no-dry-run` para borrar realmente.

### `status`

```bash
grimore status
```

Un panel con ruta de bóveda, modelos de cognición, tamaño de la DB,
estado del daemon, último escaneo y datos similares de un vistazo.
Equivalente a `/status` dentro de la shell.

### `preflight`

```bash
grimore preflight [--check-git|--no-check-git]
```

Valida la configuración, la conectividad con Ollama (incluyendo
verificación de presencia de modelos) y el acceso a la bóveda. Por
defecto requiere repo git solo si `output.auto_commit` es true —
sobrescríbelo con el flag.

### `pilot`

```bash
grimore pilot init pilot-data/study.yaml --participants 5 --study-id pilot-2026-a
grimore pilot check pilot-data/study.yaml
grimore pilot observe pilot-data/study.yaml P01 configuration \
  --outcome completed --assistance none
grimore pilot restore-backup pilot-data/study.yaml
```

`init` crea sin sobrescribir una hoja local de admisión para el piloto de dos
semanas: identificadores seudónimos, consentimiento y exactamente tres tareas
reales con criterios de éxito y método habitual. `check` bloquea una admisión
incompleta, cambios en los umbrales o campos que violen la política de no guardar
identificadores, documentos, preguntas o respuestas. `observe` registra en orden
configuración, primera respuesta, comprobación de fuente y revisión documental,
guardando solo resultado, hora y nivel de ayuda. Cada cambio deja una copia
privada reversible con `restore-backup`. No envía telemetría ni abre la bóveda.

### `daemon`

```bash
grimore daemon run         # primer plano, Ctrl-C para detener
grimore daemon start       # segundo plano, PID + log bajo el cache de la plataforma
grimore daemon stop
grimore daemon status      # muestra badge up/down
```

El daemon usa `watchdog` para observar cambios en la bóveda; un debounce
de 45 segundos agrupa guardados para que una ráfaga del editor no dispare
el LLM cinco veces. También ejecuta `[maintenance]` con la cadencia
configurada.

Las rutas PID/log se eligen vía `platformdirs`, así que las invocaciones
en primer y segundo plano siempre coinciden sobre dónde vive el estado.

### `maintenance run`

```bash
grimore maintenance run [--skip-vacuum] [--skip-purge] [--skip-checkpoint]
```

Ejecuta el pipeline de mantenimiento una vez, de inmediato. Reporta
cuántas etiquetas se depuraron, cuántos frames WAL se consolidaron y
cuántos bytes recuperó VACUUM. Cada `--skip-*` desactiva el paso
correspondiente solo para esta ejecución (los valores por defecto vuelven
la siguiente vez).

### `migrate-embeddings`

```bash
grimore migrate-embeddings <modelo-nuevo> [--status] [--abort] [--write-config/--no-write-config]
```

Intercambia en caliente el modelo de embeddings sin sacar la bóveda
de servicio. El comando construye una tabla espejo
`embeddings_migration`, re-embebe cada chunk contra el nuevo modelo
y luego reemplaza atómicamente la tabla viva en una sola
transacción. Mientras corre la migración, la búsqueda sigue
sirviendo desde los vectores antiguos.

- `--status` imprime el punto de reanudación de una migración en curso.
- `--abort` limpia la tabla espejo; los vectores vivos no se tocan.
- `--write-config` (defecto) reescribe `[cognition].model_embeddings_local`
  al terminar correctamente, para que el siguiente proceso recoja el
  nuevo modelo.

Reanudable: re-ejecutar el comando con el mismo destino retoma desde
la fila en la que el worker quedó. Un preflight rechaza arrancar si
el espacio libre en disco es menor que 2 × el tamaño actual de los
embeddings.

Reinicia cualquier daemon o shell que estuviera corriendo. El
intercambio reescribe `model_embeddings_local`, que un proceso vivo no
vuelve a leer, y su matriz de puntuación en memoria se indexa por una
firma que el intercambio deja deliberadamente igual.

El intercambio final reinserta cada fila con su `id` original, de modo
que el número de filas y el `id` máximo quedan idénticos aunque todos
los vectores hayan cambiado. La caché de matriz en disco
(`vec_matrix_cache`) no puede detectar eso por su sello, así que el
propio intercambio la borra. Si alguna vez reescribes vectores por
fuera de este comando —editando la base de datos a mano—, borra tú los
ficheros `*.vecmat.npy` y `*.vecmat.sig` que hay junto a la base de
datos, o las búsquedas seguirán puntuando contra los vectores viejos.

### `category`

```bash
grimore category list
grimore category add <ruta>
grimore category rm  <ruta> [-f|--force]
grimore category notes <ruta> [--flat]
```

Mantiene el árbol jerárquico de categorías en `taxonomy.yml`.

- `add` acepta una ruta separada por barras — los ancestros faltantes se
  crean.
- `rm` se niega a borrar una categoría que aún tenga notas asignadas
  salvo que pases `--force`. Con `--force` la categoría desaparece del
  árbol pero las notas conservan su campo `category:` (ahora colgando)
  hasta que el próximo escaneo lo reescriba.
- `notes --flat` lista solo las notas directamente bajo la ruta; sin él,
  se incluyen descendientes.

### `chronicler`

```bash
grimore chronicler list [--decay|--no-decay]
grimore chronicler check <ruta>
grimore chronicler verify <ruta>
```

Rastreo temporal de obsolescencia — marca notas que han superado su
ventana de frescura (definida por prefijo de categoría en
`[chronicler.windows]`).

- `list` muestra todo lo pasado de su ventana. `--decay` ejecuta el
  veredicto LLM cacheado en cada fila.
- `check <ruta>` ejecuta el LLM de obsolescencia contra una sola nota
  ahora (más lento, pero actual).
- `verify <ruta>` reinicia el reloj de frescura — útil cuando has
  releído una nota y confirmado que sigue siendo precisa.

Indexar o etiquetar una nota no inicia ese reloj. Hasta la primera verificación
humana, la fecha se muestra como desconocida y `list` no declara la nota vigente
ni vencida. Publicación, modificación, indexación y verificación se conservan por
separado con su procedencia.

### `mirror`

```bash
grimore mirror                                 # lista contradicciones abiertas
grimore mirror scan [-k N] [--full]            # extraer + verificar
grimore mirror show <id>                       # renderiza una en detalle
grimore mirror dismiss <id>                    # marcar como no-contradicción
grimore mirror resolve <id>                    # marcar como resuelta
```

El **Espejo Negro** extrae afirmaciones atómicas de cada nota, encuentra
pares de afirmaciones entre notas y le pregunta al LLM si se contradicen.
El predeterminado `--top-k 5` mira los cinco vecinos más cercanos de
cada afirmación; `--full` re-extrae todas las notas desde cero (lento —
reconstrucción en frío).

La extracción divide documentos largos por páginas o encabezados y después
por tramos de texto. Los límites de `[cognition]` son `claim_max_chars = 8000`
por consulta, `claim_max_sections = 8` tramos por documento y
`claim_max_claims_per_section = 10`. Se pueden elevar según el modelo y el
equipo; los tramos se eligen a lo largo de todo el documento, no solo al
principio. `mirror scan` enumera las páginas o secciones no analizadas por
presupuesto y las que fallaron. Si falla una consulta, se conservan las
afirmaciones anteriores y esa nota podrá reintentarse; un análisis parcial
por presupuesto sí se guarda, pero se declara como parcial. La bandeja de
revisión muestra estos topes como límites máximos, no como prueba de que
todas las secciones de una nota se hayan cubierto.

### `distill`

```bash
grimore distill --tag <nombre>     [-p N] [--dry-run]
grimore distill --category <ruta>  [-p N] [--dry-run]
```

Sintetiza todas las notas que llevan la etiqueta dada (o archivadas bajo
la categoría dada, recursivamente) en una sola nota de referencia bajo
`_synthesis/`. La salida recibe `grimore_generated: true` en su
frontmatter para que ejecuciones posteriores de `distill` no se incluyan
a sí mismas. El archivo también queda marcado con `synthesis_review: pending`
y conserva un informe `coverage` con notas/pasajes incluidos, límites y
coherencia de citas.

- `-p, --passages` — top-K pasajes por nota fuente (por defecto 3).
- `--dry-run` — construye la síntesis pero omite la escritura del archivo.

> **Nota:** A diferencia de `scan`/`connect`/`prune`, el `distill` del CLI
> escribe por defecto. El `/distill` de la shell refleja ese
> comportamiento, pero añade una confirmación interactiva "y/N" antes de
> cualquier escritura (ver §7).

La UI web ofrece una variante de solo vista previa en **Síntesis por tema**:
muestra el borrador, los pasajes exactos de cada fuente y advertencias de
cobertura sin escribir un archivo.

### `graph export`

```bash
grimore graph export <salida> [-f json|dot|obsidian-canvas]
                              [--suggested/--no-suggested]
                              [--suggested-top N]
                              [--suggested-threshold X]
```

Recorre el grafo de enlaces de la bóveda y lo escribe en el formato
elegido. Se producen tres tipos de aristas:

| Tipo | Origen |
|---|---|
| `wikilink` | Referencias `[[Título]]` explícitas en el cuerpo Markdown. |
| `suggested` | Top-N vecinos por coseno por nota (vectores de chunk promediados). Filtrado por `--suggested-threshold` (defecto 0.7). |
| `contradicts` | Pares de contradicciones abiertas o resueltas del **Black Mirror**, elevadas al nivel de nota. Las descartadas se excluyen. |

Formatos:

- **`json`** — `{version, nodes, edges}`. Schema estable, fácil de
  alimentar a otras herramientas.
- **`dot`** — código fuente Graphviz. Renderiza con
  `dot -Tsvg vault.dot -o vault.svg`.
- **`obsidian-canvas`** — vuelca el grafo a un `.canvas` que Obsidian
  abre directamente, con nodos agrupados por categoría en una
  cuadrícula.

El paso de vecinos semánticos es la parte lenta — pasa
`--no-suggested` para un sondeo rápido basado solo en wikilinks +
contradicciones.

### `mcp`

```bash
grimore mcp [--json/--no-json]
```

Lanza Grimore como un servidor stdio del Model Context Protocol
para que cualquier cliente MCP (Claude Desktop, Cursor, Zed, …)
pueda invocar la bóveda como herramientas. **Solo lectura** por
diseño: `grimore_ask`, `grimore_search`, `grimore_get_note`,
`grimore_connect`, `grimore_list_categories`. Las operaciones de
escritura permanecen en el CLI para que un cliente remoto nunca
pueda activar por accidente una operación destructiva.

Consulta [`docs/mcp-setup.md`](mcp-setup.md) para configuraciones
de cliente listas para copiar y la advertencia sobre el directorio
de trabajo (el servidor lee `grimore.toml` desde el `cwd`).

### `serve`

```bash
grimore serve [-H HOST] [-P PORT] [--allow-lan] [--api-token TOKEN]
              [--strict-token] [--cors-origin ORIGEN]
```

Arranca un servidor ASGI Starlette orientado a lectura, con escrituras
acotadas y protegidas por token, en
`http://127.0.0.1:8000` con una UI mínima en JS vanilla. Rutas:

A diferencia de `scan` y `ask`, `serve` permanece activo en modo de recuperación
si el preflight detecta una carpeta, modelo o servicio ausente. El panel de
configuración sigue disponible para explicar y corregir el problema; preguntas
e indexación continúan fallando de forma segura hasta cumplir sus requisitos.

| Método · Ruta | Descripción |
|---|---|
| `GET /api/health` | Versión + resumen del preflight. |
| `GET /api/setup` | Validación inicial de solo lectura: carpeta activa, modelos de chat/embeddings disponibles, efectos del escaneo y acciones de recuperación. |
| `POST /api/setup/apply` | Actualización protegida por token y confirmación explícita únicamente de carpeta y modelos de chat/embeddings. |
| `GET /api/indexing` | Progreso en vivo y de solo lectura del CLI/observador, resultados por archivo, contadores y acciones de recuperación. |
| `POST /api/indexing/start` | Inicio protegido por token de una única vista previa o indexación aplicada confirmada sobre la bóveda ya cargada. |
| `GET /api/changes` | Historial acotado de archivos creados o actualizados por operaciones aplicadas y disponibilidad de restauración. |
| `POST /api/changes/{change_id}/restore` | Restauración protegida y confirmada de las copias anteriores, rechazada si un archivo cambió después; requiere reindexar. |
| `POST /api/syntheses/preview` | Borrador de síntesis de solo lectura por etiqueta o categoría. Devuelve cuerpo, fuentes inspeccionables, cobertura y advertencias; nunca escribe el archivo propuesto. |
| `GET /api/review-inbox` | Candidatos de revisión de solo lectura que reúnen contradicciones abiertas, duplicados exactos/similares y documentos cuyo plazo de revisión venció, con pasajes inspeccionables y cobertura explícita. |
| `POST /api/review-inbox/{candidate_id}/decision` | Decisión protegida por token para confirmar, descartar o posponer de forma acotada. Actualiza el estado de revisión y su auditoría, nunca el documento fuente. |
| `POST /api/notes/{note_id}/classification` | Corrección protegida de categoría y etiquetas. Actualiza el índice, conserva la decisión humana para escaneos posteriores y no reescribe inmediatamente el documento fuente. |
| `GET /api/filter-options` | Categorías/etiquetas indexadas y formatos configurados para los controles de alcance web. |
| `POST /api/ask` | `{question, top_k?, stream?, history?, category?, tags?, formats?}`. Devuelve `answer_status`, `model_execution`, `sources` y `source_details` con pasaje, localizador, estado de versión y cobertura de extracción al indexar los PDF; streaming SSE cuando `stream: true`. |
| `POST /api/search` | `{query, top_k?, history?, category?, tags?, formats?}` → hits híbridos con fragmentos y contexto conversacional. |
| `GET /api/notes/{id}` | Metadatos + cuerpo en disco, posiciones de página/sección y cronología de publicación, modificación, indexación y verificación con procedencia. |
| `GET /api/categories` | Conteos por categoría en toda la bóveda. |
| `GET /` | La UI web. |

`category`, `tags` y `formats` estrechan `/api/ask` y `/api/search` con la
misma semántica, también en respuestas SSE. Varias etiquetas se combinan con
AND, varios formatos con OR y una categoría incluye sus descendientes. Un
alcance solicitado sin notas conserva el resultado vacío y nunca se amplía a
toda la bóveda.
La respuesta lo expone como `retrieval_scope.status = "no_matches"` y omite
por completo el embedding, la recuperación y la generación del modelo.

La UI web conserva en memoria los tres turnos completados más recientes y los
envía a ambas rutas para que respuestas y pasajes relacionados resuelvan las
mismas referencias. **Nueva conversación** limpia este contexto y los resultados
visibles sin cambiar filtros ni escribir nada en disco.

Al pulsar una cita se abre el pasaje indexado exacto, su página o sección y el
estado de verificación actual/modificado/ausente. Los documentos vigentes
resaltan el pasaje dentro de su contexto y permiten navegar por página o
sección; los demás estados conservan la evidencia indexada con una advertencia.

El selector del encabezado cambia toda la interfaz entre español e inglés y
recuerda la elección en ese navegador. Las categorías, los pasajes relacionados,
las glosas marginales, las citas y los enlaces de fuentes son accesibles con
teclado y se activan con **Enter** o **Espacio**. **Ctrl/Cmd + Enter** envía la
pregunta. Los visores de fuentes mantienen el foco dentro mientras están
abiertos, se cierran con **Escape** y devuelven el foco al control que los abrió.
Los errores de validación, red, búsqueda y respuesta aparecen junto a la
pregunta, se anuncian a tecnologías de asistencia y no fallan en silencio.

Compuertas de seguridad codificadas en el CLI (no solo en la doc):

- Por defecto enlaza solo a loopback. Establecer `--host 0.0.0.0` (o
  cualquier dirección no-loopback) exige **ambos** `--allow-lan` y
  `--api-token`.
- `--api-token` (también leído desde `GRIMORE_API_TOKEN` — prefiere la
  variable de entorno: un token escrito en la línea de comandos es
  visible para otros usuarios locales vía `ps`) se valida en **cada**
  petición `/api/*` de un cliente no-loopback, GET incluido. La
  comparación es de tiempo constante, y los fallos repetidos desde una
  misma dirección se limitan (HTTP 429) para acotar la fuerza bruta.
- Los clientes loopback saltan el token en operaciones de lectura para que la
  UI local funcione sin credenciales en cada petición. Las dos rutas de
  escritura siempre lo exigen, también desde loopback; la UI conserva el token
  introducido únicamente en memoria de la pestaña. En Android/Termux la
  exención de lectura es más débil de lo que parece —
  *cualquier app del dispositivo* puede alcanzar los puertos de
  localhost — así que allí pasa `--strict-token` para exigir el token
  también a los clientes loopback. En modo estricto, introduce el token en la
  UI o usa cabeceras `Authorization: Bearer` explícitas.
- CORS está desactivado por defecto. `--cors-origin <origen>` añade
  exactamente un origen (sin comodines).

El extra `serve` solo trae `starlette` + `uvicorn` para que la
instalación siga siendo segura en Termux (no hace falta la wheel
de pydantic-core en ARM).

---

## 7. La shell interactiva — `grimore shell`

La shell es la interfaz recomendada para el día a día. Mantiene un
`Session` cálido, por lo que los `ask` consecutivos reutilizan el
embebedor y el router LLM sin pagar costes de arranque en frío.

Ábrela:

```bash
grimore shell
```

Aterrizas en un banner y el prompt `❯ `. Escribe una pregunta y pulsa
Enter — no necesitas el prefijo `/ask`:

```text
❯ qué temas atraviesan mis notas sobre nihilismo?
…transmite la respuesta del Oráculo en tiempo real…
[[camus-revolt]] · [[nietzsche-gay-science]] · [[cioran-bitter-cradle]]
```

### Composición de la entrada

- **Enter sin más** envía.
- **`Esc` y luego `Enter`** *o* **`Alt+Enter`** insertan un salto de línea
  (para preguntas multi-párrafo o pegar un bloque).
- Las líneas que terminan en `\` continúan en la siguiente y se unen en
  una sola entrada lógica antes del dispatch.
- **`Ctrl+C`** cancela la entrada actual *o* la pregunta en curso sin
  matar el bucle.
- **`Ctrl+D`** (EOF) sale de la shell limpiamente.
- **`Ctrl+R`** (Emacs) abre búsqueda inversa en el historial.

### Comandos slash

Cada verbo del CLI tiene un gemelo con slash, más un puñado de ayudantes
exclusivos de la shell. Los comandos slash autocompletan en un popup en
cuanto escribes `/`.

| Slash | Qué hace |
| :--- | :--- |
| `/ask` | Pregunta explícita — normalmente innecesaria (simplemente escribe la pregunta). |
| `/scan`, `/connect`, `/prune` | Reflejan verbos del CLI, con confirmaciones. |
| `/status`, `/tags`, `/preflight` | Inspección de solo lectura. |
| `/category list \| add \| rm \| notes` | Igual que el CLI. |
| `/chronicler list \| check \| verify` | Igual que el CLI. |
| `/mirror`, `/mirror scan/show/dismiss/resolve` | Igual que el CLI. |
| `/distill` | Igual que el CLI, más una confirmación al escribir. |
| `/again` | Re-formula la pregunta anterior con los mismos flags. |
| `/why` | Re-imprime las fuentes citadas por la última respuesta. |
| `/pin @nota […]` | Fija notas a cada futura pregunta (`/pin` solo lista los pins). |
| `/unpin [@nota]` | Quita un pin (o todos, si se llama sin argumentos). |
| `/save [ruta] [-f]` | Exporta la transcripción de la sesión como nota en la bóveda. |
| `/history [N]` | Muestra las últimas N preguntas (por defecto 10). |
| `/forget` | Vacía la memoria conversacional (turnos + last_*). Los pins se conservan. |
| `/thread save\|resume\|list` | Persiste / recarga conversaciones entre shells. |
| `/resume <nombre>` | Atajo para `/thread resume <nombre>`. |
| `/threads` | Atajo para `/thread list`. |
| `/models [chat\|embed [nombre\|índice]]` | Lista modelos del backend y cambia en vivo. |
| `/refresh` | Vacía los servicios cacheados + el índice de menciones. |
| `/clear` | Limpia la pantalla. |
| `/help [cmd]` | Lista los comandos, o detalla uno. |
| `/exit`, `/quit` | Sale. |

Un error tipográfico como `/scna` dispara una sugerencia
`¿Quisiste decir: /scan?` vía `difflib`.

### Menciones con `@`

Un token que empieza con `@` se resuelve contra el índice de la bóveda y
el cuerpo completo de la nota correspondiente se adjunta a la próxima
pregunta como contexto prioritario:

```text
❯ @camus-revolt cómo conecta esto con el absurdismo?
```

Orden de resolución:

1. ruta exacta bajo la raíz de la bóveda (con o sin `.md`),
2. coincidencia exacta de título (insensible a mayúsculas),
3. mejor match `rapidfuzz` por encima del umbral configurado.

Cada resolución se re-valida mediante
`SecurityGuard.resolve_within_vault`, por lo que un intento como
`@../escape` es rechazado y el `@token` literal permanece en el texto del
mensaje. Los tokens que no resuelven generan una línea muda de "ninguna
nota coincidió" y se dejan como texto plano en la pregunta.

Las adjunciones están limitadas a 32.000 caracteres por nota (el
`EMBED_MAX_CHARS` del embebedor).

El completador ofrece autocompletado fuzzy para tokens `@`; la columna
meta a la derecha muestra la puntuación de coincidencia, así puedes
ajustar `[shell] fuzzy_threshold` a tu gusto.

### Notas fijadas (pins)

Las menciones `@` son de un solo uso. Para adjuntar una nota a *todas*
las preguntas futuras de la sesión, fíjala:

```text
❯ /pin @nietzsche-gay-science
Pinned: [[nietzsche-gay-science]]

❯ /pin                            # lista los pins actuales
❯ /unpin @nietzsche-gay-science   # quita uno
❯ /unpin                          # quita todos
```

El segmento `pins: N` de la barra inferior se actualiza al instante.

### Confirmaciones de seguridad

Cualquier comando que escriba en la bóveda confirma antes de actuar:

```text
❯ /scan --no-dry-run
scan --no-dry-run will write frontmatter to every changed note.
Continue? [y/N] _
```

Pasa `--yes` para saltarte la confirmación — útil para scripts:

```text
❯ /scan --no-dry-run --yes
```

Comandos que confirman por defecto:

- `/scan --no-dry-run`
- `/connect --no-dry-run`
- `/prune --no-dry-run`
- `/category rm`
- `/distill` (cuando no está en `--dry-run`)

### Guardar transcripciones

```text
❯ /save                                       # ruta por defecto: _transcripts/<ts>.md
❯ /save reflexiones/oraculo-2026-05-20.md
❯ /save reflexiones/oraculo-2026-05-20.md --force   # sobrescribir uno existente
```

La transcripción contiene un encabezado `Q1.` `Q2.` … por pregunta más el
cuerpo completo de la respuesta más reciente (con sus fuentes). La ruta
se re-valida vía `SecurityGuard.resolve_within_vault`, así que las rutas
que escapan de la bóveda son rechazadas. Si el archivo destino ya existe,
`/save` se niega a sobrescribirlo salvo que pases `-f` / `--force`.

### Persistencia de conversaciones — `/thread`, `/resume`, `/threads`

`/save` exporta una transcripción Markdown puntual dentro de la
bóveda. Para persistir un *hilo* de conversación *retomable* usa en
su lugar el namespace `/thread`. Los hilos sobreviven al reinicio de
la shell, así que una sesión de investigación larga puede retomarse
más tarde.

```text
❯ /thread save                                # slug automático desde la primera pregunta
❯ /thread save investigacion-sobre-estoicismo # slug explícito
❯ /thread resume investigacion-sobre-estoicismo  # recárgalo
❯ /thread list                                # muestra todos los hilos guardados
❯ /threads                                    # atajo para /thread list
❯ /resume investigacion-sobre-estoicismo      # atajo para /thread resume …
```

Qué se guarda:

- Un archivo JSONL por hilo bajo `shell.threads_dir` (defecto
  `~/.grimore/threads/`).
- Cada línea es un turno: `{ts, q, a, sources}`.
- Escritura atómica — el archivo se escribe a `<slug>.jsonl.tmp` y
  luego se renombra, así que un crash a mitad de escritura no puede
  dejar un hilo corrupto.

Al retomar se rellenan `last_question` y `last_answer` desde el
último turno para que `/again` y `/why` funcionen de inmediato. La
memoria conversacional rodante (últimos 3 turnos, ver abajo) se
restaura desde el archivo, así que los follow-ups como "expande el
último punto" resuelven limpiamente.

El slug para un `/thread save` sin argumentos se deriva de las
primeras seis palabras de la primera pregunta; los caracteres
no-ASCII se reducen a ASCII y la puntuación se elimina para que el
nombre de archivo siga siendo portable.

### Memoria conversacional y `/forget`

Dentro de una misma sesión de shell, Grimore conserva una memoria
rodante de los últimos tres turnos (`{q, a, sources}` cada uno). La
usa en dos sitios:

- **Reescritura de consulta para recuperación.** Pronombres y
  referencias resuelven contra turnos anteriores ("expande sobre eso"
  encuentra las mismas notas).
- **Coherencia de respuesta.** Un bloque corto "Conversación
  reciente" precede al contexto recuperado para que el modelo
  permanezca en el tema entre turnos.

El `grimore ask` puntual del CLI nunca alimenta esta memoria — su
comportamiento es idéntico al de v2.0. Dentro de la shell, `/forget`
descarta la memoria más `last_question` / `last_answer` / el
question log sin tocar los pins ni el historial de shell en disco:

```text
❯ /forget
Conversación olvidada — empezando desde cero.
```

`/refresh` (que reconstruye los cachés tras un scan desde otra
terminal) también limpia implícitamente la memoria conversacional.

### Cambiar de modelo en vivo

```text
❯ /models                       # lista modelos Ollama instalados + selección actual
❯ /models chat ministral-3:14b  # cambia el modelo de chat (en vivo y persistido)
❯ /models embed embeddinggemma  # cambia el modelo de embeddings
```

Los cambios de modelo se persisten de vuelta en `[cognition]` de
`grimore.toml`. El cambio de chat es instantáneo; un cambio de embeddings
descarta el embebedor cacheado para que la siguiente consulta lo
reconstruya con el modelo nuevo.

También puedes pasar un índice en vez de un nombre:

```text
❯ /models chat 2
```

### Barra inferior

El pie persistente muestra el estado vivo de la sesión:

```text
vault: Biblioteca  •  chat: qwen2.5:3b  •  embed: nomic-embed-text  •  dry-run  •  pins: 2
```

Se re-renderiza con cada tecla — `/models` y `/pin` se reflejan al
instante.

### Modo vi

Activa `[shell] vi_mode = true` en `grimore.toml` para habilitar el modo
de edición modal vi de prompt_toolkit. El glifo del prompt pasa de `❯ ` a
`∙ ` cuando estás en modo normal.

---

## 8. Taxonomía: `taxonomy.yml`

Coloca un `taxonomy.yml` en la raíz de tu bóveda para fijar un
vocabulario controlado y un árbol de categorías:

```yaml
vocabulary:
  - filosofia
  - ocultismo-clasico
  - nihilismo

categories:
  Historia:
    - Antigua
    - Moderna
  Ciencia:
    Fisica:
      - Cuantica
    - Biologia
```

- **Normalización de etiquetas** — `"Ocultismo Clásico"` →
  `ocultismo-clasico`. Las etiquetas conocidas se reescriben a su forma
  canónica; las desconocidas pasan tal cual (el etiquetador puede
  introducir nuevas, pero también puedes sembrar el vocabulario para
  sesgarlo).
- **Las categorías son mutuamente excluyentes**: cada nota termina
  archivada bajo exactamente una ruta jerárquica canónica.
- **La resolución es insensible a acentos y mayúsculas**: `"física"` y
  `"Fisica"` ambos resuelven a `Ciencia/Física` una vez que esa ruta
  existe.
- Un `taxonomy.yml` faltante o malformado recae en los valores por
  defecto — el ingest nunca se bloquea.

---

## 9. Convenciones de frontmatter

Grimore lee (y selectivamente escribe) los siguientes campos de
frontmatter YAML. Cualquier cosa que añadas a mano se preserva en el
ciclo lectura/escritura.

```yaml
---
title: "Camus y la Rebelión"
tags: [filosofia, nihilismo, absurdismo]
summary: |
  Párrafo corto que se reescribe en cada scan.
category: Filosofia/Existencialismo
last_tagged: "2026-05-20T10:14:22Z"
privacy: never_process     # opcional — excluye de la cognición por completo
grimore_generated: true    # escrito por las salidas de `distill`; las protege de re-destilación
---
```

- `last_tagged` es una marca temporal UTC ISO del procesamiento automático; no
  equivale a publicación, modificación ni verificación. **Chronicler** solo
  inicia el reloj tras `chronicler verify`.
- `privacy: never_process` es el único campo verificado *antes* de
  cualquier llamada al LLM, así que realmente mantiene la nota fuera del
  cable.
- Para **documentos no-Markdown** (PDF, EPUB, DOCX, …) los mismos
  campos viven en el sidecar `.md` bajo `<bóveda>/<sidecar_dir>/` en
  lugar del archivo fuente. El binario original nunca se modifica. Para
  marcar `privacy: never_process` en un PDF, escribe el frontmatter en
  su sidecar — Grimore preserva las ediciones manuales entre re-scans.

---

## 10. Privacidad y seguridad

**Local por construcción.** Con `cognition.allow_remote = false` (el
predeterminado), toda llamada al LLM es rechazada salvo que el endpoint
resuelva a una dirección loopback — y la conexión se fija (pin) a esa IP
validada (el hostname original viaja en la cabecera `Host`), de modo que un
nombre que pasa la comprobación no puede reasignarse (rebind) a otra
dirección antes de que la petición llegue. Cambia el flag solo si estás
enrutando a sabiendas a una máquina de confianza por `https`.

**Salida por nota.** `privacy: never_process` excluye una nota de la
cognición por completo (ver §9).

**Detección de PII.** Antes de cualquier llamada al LLM, el contenido se
escanea en busca de claves de API, correos, IPs y claves SSH; las
coincidencias se marcan como WARNING en el log estructurado (la llamada
igual procede — Grimore no descarta silenciosamente tu trabajo, pero te
enteras).

**Endurecimiento contra prompt-injection.** Los marcadores de rol
(`### System:` y similares) y los tokens de plantilla de chat se
neutralizan antes de llegar al LLM — aplicado tanto al contenido de notas
recuperado como a cualquier historial de conversación que pase un cliente
(el campo `history` de la API HTTP), de modo que ninguno puede colar
instrucciones al modelo.

**Parseo de documentos endurecido.** Los archivos EPUB/DOCX/ODT son XML por
dentro; Grimore parsea esos miembros con defensas contra expansión de
entidades y entidades externas (un parser basado en `defusedxml`, con un
pre-rechazo por bytes de `DOCTYPE`/`ENTITY` como respaldo). Un documento
malicioso tipo «billion-laughs» soltado en una bóveda vigilada se rechaza,
en lugar de agotar la memoria.

**Git Guard.** Cada mutación va precedida de un auto-commit. `git reflog`
es tu deshacer.

**Backups diarios.** El daemon hace snapshots de SQLite y conserva los
últimos cinco bajo `backups/` con `chmod 0700`/`0600`.

> `backups/` contiene **copias SQLite crudas y sin cifrar** que incluyen
> los primeros 500 caracteres de cada fragmento embebido. Si tu bóveda
> contiene secretos, móntala sobre un FS cifrado (`gocryptfs`, `age`,
> LUKS). Grimore deliberadamente no cifra en reposo — el cifrado de
> disco completo es la frontera correcta para una herramienta
> mono-usuario local-first.

**Confinamiento de rutas.** Cada archivo que Grimore toca (notas,
transcripciones, adjunciones de menciones `@`) se re-resuelve mediante
`SecurityGuard.resolve_within_vault`. Un token como `@../escape` no puede
exfiltrar un archivo fuera de la bóveda, ni siquiera vía enlace
simbólico.

---

## 11. Integraciones — MCP, API HTTP, backends compatibles con OpenAI

Tres superficies permiten usar Grimore fuera del terminal.

### Servidor MCP

`grimore mcp` expone la bóveda como herramientas a cualquier cliente
del Model Context Protocol (Claude Desktop, Cursor, Zed). Es un
servidor stdio — el cliente lanza `grimore mcp` y habla JSON-RPC por
su stdin/stdout.

El servidor MCP es **solo lectura**: `grimore_ask`, `grimore_search`,
`grimore_get_note`, `grimore_connect`, `grimore_list_categories`. Los
escaneos y migraciones permanecen en el CLI.

Los resultados cruzan hacia el cliente MCP: `grimore_ask` incluye los
pasajes indexados completos usados como contexto, mientras
`grimore_get_note` devuelve el cuerpo actual completo de una nota elegida.
Para un PDF citado, `source_details[].indexed_extraction` indica si faltaba
texto al indexarlo; el estado `supported` de una respuesta no elimina ese
aviso de cobertura.
La búsqueda solo devuelve vistas previas de 200 caracteres; conexión y
categorías usan datos almacenados sin invocar un modelo. La tabla de frontera
por herramienta en [`docs/mcp-setup.md`](mcp-setup.md) también enumera qué
envía Grimore a sus servicios de chat y embeddings.

Configuraciones de cliente listas para copiar y la advertencia sobre
el directorio de trabajo (el servidor lee `grimore.toml` desde el
`cwd`, y los clientes MCP lanzan desde otro lugar) están en
[`docs/mcp-setup.md`](mcp-setup.md).

### API HTTP + UI web

`grimore serve` arranca un servidor ASGI Starlette en
`127.0.0.1:8000` con una UI mínima en JS vanilla. Solo loopback por
defecto; los binds no-loopback exigen tanto `--allow-lan` como
`--api-token`.

```bash
# Solo loopback — ábrelo en tu navegador
grimore serve

# Exponerlo en la LAN con un bearer token. La variable de entorno
# mantiene el token fuera de `ps` — `serve` avisa si lo ve en los
# argumentos.
GRIMORE_API_TOKEN="$(openssl rand -hex 32)" \
  grimore serve --host 0.0.0.0 --port 8000 --allow-lan
```

Las respuestas en streaming usan Server-Sent Events en
`POST /api/ask` cuando el cuerpo lleva `stream: true`. CORS está
desactivado por defecto; pasa `--cors-origin <origen>` para habilitar
exactamente un origen.

La UI incluida no depende de fuentes ni scripts remotos. Su indicador de
salud muestra la ejecución de modelos como `local`, `mixed` o `remote`;
`grimore status`, `GET /api/health` y las respuestas exponen además la misma
separación entre chat y embeddings. Consultar este modo no resuelve DNS ni
abre conexiones.

El botón **Setup** abre un asistente inicial. Valida la carpeta
ya cargada por el servidor, enumera los modelos reportados por los servicios de
chat y embeddings, explica los efectos del escaneo en vista previa y aplicado,
y genera valores TOML para copiar. Si `serve` tiene token de API, el mismo panel
puede guardar atómicamente solo esos tres campos tras autenticar el bearer. El
registro de indexación puede iniciar una vista previa o un escaneo aplicado con
confirmación separada, siempre sobre la bóveda fijada al arrancar el servidor y
sin ejecuciones concurrentes. Guardar cambios de configuración exige reiniciar
antes de indexar. La vista previa expone valores anteriores y propuestos sin
escribir; las operaciones aplicadas conservan copias privadas recuperables y el
panel permite restaurarlas si ningún archivo cambió después.

Cuando se define un token, este protege **todas** las rutas `/api/*` —
incluidos los GET, no solo los POST al LLM — para cualquier cliente
no-loopback; las lecturas loopback siguen abiertas para que la UI web local
funcione sin token. Las rutas de escritura acotadas siempre exigen el bearer. El par
(peer) se lee de la propia conexión, así que un
`X-Forwarded-For` falsificado no puede saltarse la comprobación, el token
se compara en tiempo constante, y los fallos repetidos desde una misma
dirección se limitan (HTTP 429) para acotar la fuerza bruta. Un `top_k` en
el cuerpo de la petición se acota a un techo razonable (y un valor no
numérico se rechaza con `400`) en lugar de amplificar el trabajo de
recuperación.

En Android/Termux, desconfía de la exención loopback: Android no aísla
localhost entre apps, así que *cualquier app del dispositivo* puede
alcanzar un puerto `127.0.0.1` a la escucha. Pasa allí `--strict-token`
para exigir el bearer token también a los clientes loopback; introdúcelo en la
UI o usa cabeceras `Authorization: Bearer` explícitas.

El extra `serve` es obligatorio: `pip install 'grimore[serve]'`. En
Linux/Windows arrastra FastAPI como ruta de upgrade; la
implementación enviada usa solo Starlette + uvicorn para que una
instalación en Termux siga siendo viable (pydantic-core no tiene
wheel preconstruida para Termux/ARM).

### Backends LLM compatibles con OpenAI

Grimore puede hablar con cualquier servidor que implemente la forma
`POST /v1/chat/completions` de OpenAI — llama.cpp server, vLLM, LM
Studio, OpenRouter, OpenAI propiamente. Configúralo en `[cognition]`:

```toml
[cognition]
llm_backend     = "openai"
llm_base_url    = "http://localhost:8080/v1"   # por defecto en llama.cpp server
llm_api_key_env = "GRIMORE_LLM_API_KEY"        # variable con el token
model_llm_local = "nombre-de-tu-modelo"        # el que reporte el servidor
allow_remote    = false                        # déjalo en false para mantener loopback
```

Después, en la shell:

```bash
export GRIMORE_LLM_API_KEY="sk-…"   # déjala vacía para servidores sin autenticación
grimore preflight
grimore ask "pregunta de calentamiento"
```

Notas:

- El mismo `SecurityGuard.validate_llm_host` enforza
  `allow_remote = false` contra tu `llm_base_url`. Para apuntar a un
  host remoto debes establecer explícitamente `allow_remote = true`
  *y* `llm_base_url` debe usar `https://`.
- El bearer token se lee **diferidamente** desde la variable de
  entorno en cada llamada, así que un token rotado surte efecto sin
  reiniciar Grimore.
- El modo JSON usa `response_format: {"type": "json_object"}`; los
  servidores que no lo honran caen al regex de "extrae el primer
  `{…}`" del router — la misma vía de respaldo usada en la rama de
  Ollama.
- Los embeddings siguen yendo por Ollama. El switch `llm_backend`
  solo afecta a la ruta de chat-completion.

---

## 12. Trabajar con modelos grandes

Los modelos más grandes (p. ej. `ministral-3:14b`) suelen necesitar una
ventana de calentamiento más larga antes del primer token. Sube los
timeouts relevantes en `grimore.toml`:

```toml
[cognition]
model_llm_local   = "ministral-3:14b"
request_timeout_s = 600
stream_timeout_s  = 900
```

**Cómo dimensionarlos en vez de adivinar.** El coste lo domina el *prompt
eval*, no la generación, y escala con el tamaño de la entrada. Mide tu
combinación real de modelo y máquina una vez:

```bash
curl -s http://127.0.0.1:11434/api/generate -d '{
  "model": "TU_MODELO", "prompt": "...", "stream": false
}' | python3 -c 'import json,sys; d=json.load(sys.stdin); \
print({k: d[k] for k in ("total_duration","prompt_eval_count", \
"prompt_eval_duration","eval_count","eval_duration") if k in d})'
```

Divide `prompt_eval_count / prompt_eval_duration` para obtener tokens/s de
lectura. `stream_timeout_s` debe cubrir el prompt eval **completo**: con
`stream: true` Ollama no emite el primer token hasta terminarlo, así que un
límite por debajo aborta sin haber recibido nada — y el síntoma es un
panel "No answer", no un error de timeout. Como el Oracle envía hasta `context_max_chars` de contexto,
`stream_timeout_s` casi siempre tiene que ser el mayor de los dos.

Si el resultado te sale absurdo (varios minutos por nota), el timeout no es
el arreglo: baja `context_max_chars` o usa un modelo más pequeño para el
tagger. Un timeout más alto solo hace que esperes más antes del mismo
resultado.

Algunos modelos de razonamiento (`qwen3.5:0.8b`, familia `deepseek-r1`,
etc.) emiten una fase de "thinking": Ollama transmite fragmentos con
`response=""` vacío y el contenido real vive en `thinking="…"` durante
hasta un minuto antes del primer token visible para el usuario. La shell
muestra un spinner durante esta fase y no parecerá congelada.

`Ctrl+C` durante la espera cancela limpiamente y te devuelve al prompt.

---

## 13. Solución de problemas

**`preflight` dice que Ollama es inalcanzable.**
Ollama no está corriendo, o `allow_remote = false` está rechazando tu
hostname. Ejecuta `ollama serve` en otra terminal y reintenta.

**`preflight` dice que un modelo no está descargado.**
`ollama pull qwen2.5:3b` (o el que indique tu config). Luego vuelve a
ejecutar `grimore preflight`.

**`ask` devuelve respuestas "sin fuentes".**
La bóveda aún no se ha embebido. Ejecuta `grimore scan --no-dry-run`
primero. Para notas recién añadidas, el debounce de 45 segundos del
daemon aún no ha disparado — espera o ejecuta `scan` manualmente.

**`/save` reporta un rechazo por travesía de bóveda.**
La ruta resuelve fuera de la raíz de la bóveda. Usa una ruta relativa a
la bóveda (p. ej. `_transcripts/foo.md`), no una ruta absoluta.

**`@<título>` no autocompleta.**
El umbral fuzzy puede estar demasiado alto. Baja `[shell] fuzzy_threshold`
(el valor por defecto es 55; prueba 35–45). `/refresh` reconstruye el
índice cacheado de la bóveda por si el título se añadió después de abrir
la shell.

**La shell parece congelada después de hacer una pregunta.**
El modelo está en su fase de thinking. El spinner debería ser visible;
si no lo está, tu terminal podría estar comiéndose códigos de escape.
Los modelos grandes pueden tardar 30–90 segundos antes del primer token.

**`scan` falla con "git not initialised".**
Ejecuta `git init` dentro de la bóveda, o establece
`output.auto_commit = false` en `grimore.toml` (pierdes la red de
seguridad).

---

## 14. Glosario

- **Bóveda (vault)** — el directorio raíz que contiene tus notas Markdown.
- **Ingest** — el observador watchdog que detecta cambios.
- **Cognición** — el pase impulsado por LLM de etiquetas, categoría,
  resumen y embeddings.
- **Memoria** — la base de datos SQLite (WAL + FTS5 + columnas vectoriales).
- **Oráculo** — la capa RAG; responde preguntas con citas.
- **Síntesis** — `connect`, `distill`, el bloque de conexiones sugeridas.
- **Chronicler** — rastreador temporal de obsolescencia.
- **Espejo Negro** — detector de contradicciones entre notas.
- **RRF** — Reciprocal Rank Fusion, la estrategia de fusión BM25 + coseno.
- **Dry-run** — modo previsualización; no se escribe a disco.
- **Pin** — una nota adjuntada a cada pregunta de la sesión actual.
- **Mención `@`** — una nota adjuntada de un solo uso a la próxima pregunta.
- **Hilo** — una transcripción JSONL de una conversación de shell
  guardada bajo `shell.threads_dir`, retomable entre sesiones.
- **Perfil** — un bloque `[profiles.<nombre>]` en `grimore.toml` que
  se fusiona profundamente con los valores por defecto cuando se
  activa con `--profile` o `GRIMORE_PROFILE`.
- **MCP** — Model Context Protocol; el contrato JSON-RPC sobre stdio
  con el que Grimore se enchufa a Claude Desktop / Cursor / Zed.
- **Backend** — el servidor de chat-completion al que Grimore le
  habla. `[cognition].llm_backend = "ollama" | "openai"` decide
  entre `/api/generate` de Ollama y `/v1/chat/completions`
  compatible con OpenAI.

---

Publicado bajo la Licencia MIT.
