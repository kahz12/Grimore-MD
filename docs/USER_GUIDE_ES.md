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
6. [Comandos del día a día](#6-comandos-del-día-a-día)
   - [`scan`](#scan)
   - [`connect`](#connect)
   - [`ask`](#ask)
   - [`tags`](#tags)
   - [`prune`](#prune)
   - [`status`](#status)
   - [`preflight`](#preflight)
   - [`daemon`](#daemon)
   - [`maintenance run`](#maintenance-run)
   - [`category`](#category)
   - [`chronicler`](#chronicler)
   - [`mirror`](#mirror)
   - [`distill`](#distill)
7. [La shell interactiva — `grimore shell`](#7-la-shell-interactiva--grimore-shell)
   - [Composición de la entrada](#composición-de-la-entrada)
   - [Comandos slash](#comandos-slash)
   - [Menciones con `@`](#menciones-con-)
   - [Notas fijadas (pins)](#notas-fijadas-pins)
   - [Confirmaciones de seguridad](#confirmaciones-de-seguridad)
   - [Guardar transcripciones](#guardar-transcripciones)
   - [Cambiar de modelo en vivo](#cambiar-de-modelo-en-vivo)
   - [Barra inferior](#barra-inferior)
   - [Modo vi](#modo-vi)
8. [Taxonomía: `taxonomy.yml`](#8-taxonomía-taxonomyyml)
9. [Convenciones de frontmatter](#9-convenciones-de-frontmatter)
10. [Privacidad y seguridad](#10-privacidad-y-seguridad)
11. [Trabajar con modelos grandes](#11-trabajar-con-modelos-grandes)
12. [Solución de problemas](#12-solución-de-problemas)
13. [Glosario](#13-glosario)

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
  `cognition.request_timeout_s` a 180+ antes de usarlo.

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
allow_remote           = false  # bloquea endpoints de Ollama que no sean loopback
hybrid_search          = true   # BM25 + coseno vía RRF
rrf_k                  = 60     # curva de peso por rango; menor = más pronunciada
connect_threshold      = 0.7    # umbral coseno para sugerir un wikilink
request_timeout_s      = 60     # /api/generate, vía JSON
stream_timeout_s       = 120    # /api/generate, vía streaming
embed_timeout_s        = 30     # /api/embeddings

[ingest]
# Motor de PDF. "pypdf" es el predeterminado siempre disponible;
# "pdfplumber" y "pymupdf" son extras opcionales que preflight reporta.
# PyMuPDF es AGPL — verifica la compatibilidad de licencia antes de
# usarlo.
pdf_engine    = "pypdf"
# OCR como respaldo para PDFs escaneados (páginas sin capa de texto).
# Desactivado por defecto; requiere el extra `ocr` y el binario
# `tesseract` en el PATH.
ocr           = false
ocr_timeout_s = 30
# Sniffer de magic-bytes para archivos sin extensión o mal nombrados.
# Desactivado por defecto; requiere el extra `sniff` (python-magic) y
# libmagic del sistema.
sniff_magic   = false

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
enabled    = false
log_events = true

[shell]
vi_mode         = false  # true para activar modo vi en prompt_toolkit
fuzzy_threshold = 55     # 0–100, puntuación rapidfuzz mínima para autocompletado @
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

- **Motores de PDF alternativos.** `pdfplumber` maneja columnas y
  tablas mejor que `pypdf`; `pymupdf` (AGPL — verifica compatibilidad
  de licencia) tiene la mejor calidad de extracción. Instala el extra
  correspondiente:

  ```bash
  pip install 'grimore[pdf-plumber]'
  pip install 'grimore[pdf-mupdf]'   # AGPL-3.0
  ```

  Luego configura `pdf_engine` como `"pdfplumber"` o `"pymupdf"`. El
  cambio se aplica en el siguiente scan sin reiniciar.

- **OCR para PDFs escaneados.** Las páginas con una capa de texto vacía
  pueden rasterizarse y procesarse con OCR vía `tesseract`. Se
  necesitan dos dependencias: las wheels de Python
  (`pip install 'grimore[ocr]'`) y el binario `tesseract` instalado en
  el sistema. Luego activa `ocr = true`. Las secciones generadas por
  OCR llevan el encabezado sintético `(ocr)` para que los revisores
  puedan auditarlas.

- **Sniffer de magic-bytes.** Cuando `sniff_magic = true`, los archivos
  cuya extensión falta o no coincide con `formats` se inspeccionan por
  su tipo de contenido real (vía `libmagic`) y se enrutan al adaptador
  correcto si existe. Útil para bóvedas con descargas ad-hoc. Instala
  el extra `sniff` y `libmagic` del sistema (Linux:
  `apt install libmagic1`; Termux: `pkg install file`; Windows:
  `pip install python-magic-bin`).

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
extracción; solo los documentos cuyos bytes han cambiado pagan la
re-extracción. Dentro de eso, un `content_hash` sin cambios omite la
llamada al LLM por completo. Reescanear una bóveda totalmente indexada
es prácticamente gratis incluso para una biblioteca de PDFs de 500 MB.

**Qué se escribe.** Con `--no-dry-run`:

- `tags`, `summary`, `category`, `last_tagged` se escriben en el
  frontmatter YAML de cada nota,
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

### `tags`

```bash
grimore tags [-n N]
```

Tabla de frecuencia de todas las etiquetas en uso. `-n` limita las filas
mostradas (por defecto 30).

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

### `distill`

```bash
grimore distill --tag <nombre>     [-p N] [--dry-run]
grimore distill --category <ruta>  [-p N] [--dry-run]
```

Sintetiza todas las notas que llevan la etiqueta dada (o archivadas bajo
la categoría dada, recursivamente) en una sola nota de referencia bajo
`_synthesis/`. La salida recibe `grimore_generated: true` en su
frontmatter para que ejecuciones posteriores de `distill` no se incluyan
a sí mismas.

- `-p, --passages` — top-K pasajes por nota fuente (por defecto 3).
- `--dry-run` — construye la síntesis pero omite la escritura del archivo.

> **Nota:** A diferencia de `scan`/`connect`/`prune`, el `distill` del CLI
> escribe por defecto. El `/distill` de la shell refleja ese
> comportamiento, pero añade una confirmación interactiva "y/N" antes de
> cualquier escritura (ver §7).

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
| `/models [chat\|embed [nombre\|índice]]` | Lista modelos Ollama y cambia en vivo. |
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

- `last_tagged` es una marca temporal UTC ISO; el **Chronicler** la usa
  para calcular obsolescencia.
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
predeterminado), toda llamada a Ollama es rechazada salvo que el endpoint
resuelva a una dirección loopback. Cambia el flag solo si estás enrutando
a sabiendas a una máquina LAN de confianza.

**Salida por nota.** `privacy: never_process` excluye una nota de la
cognición por completo (ver §9).

**Detección de PII.** Antes de cualquier llamada al LLM, el contenido se
escanea en busca de claves de API, correos, IPs y claves SSH; las
coincidencias se marcan como WARNING en el log estructurado (la llamada
igual procede — Grimore no descarta silenciosamente tu trabajo, pero te
enteras).

**Endurecimiento contra prompt-injection.** Los marcadores de rol
(`### System:` y similares) incrustados dentro del contenido de las
notas se neutralizan antes de llegar al LLM.

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

## 11. Trabajar con modelos grandes

Los modelos más grandes (p. ej. `ministral-3:14b`) suelen necesitar una
ventana de calentamiento más larga antes del primer token. Sube los
timeouts relevantes en `grimore.toml`:

```toml
[cognition]
model_llm_local   = "ministral-3:14b"
request_timeout_s = 180
stream_timeout_s  = 240
```

Algunos modelos de razonamiento (`qwen3.5:0.8b`, familia `deepseek-r1`,
etc.) emiten una fase de "thinking": Ollama transmite fragmentos con
`response=""` vacío y el contenido real vive en `thinking="…"` durante
hasta un minuto antes del primer token visible para el usuario. La shell
muestra un spinner durante esta fase y no parecerá congelada.

`Ctrl+C` durante la espera cancela limpiamente y te devuelve al prompt.

---

## 12. Solución de problemas

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

## 13. Glosario

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

---

Publicado bajo la Licencia MIT.
