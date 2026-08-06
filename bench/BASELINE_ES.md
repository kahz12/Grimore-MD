# Grimore — Línea base de rendimiento

*[English version](BASELINE_EN.md)*

> Números de partida contra los que se comparan las optimizaciones de rendimiento. Cada
> optimización posterior añade una columna "después" a las tablas de abajo.
>
> Sobre las referencias cruzadas: las optimizaciones se numeran (opt. 1–10) y los cuellos de botella
> se nombran (B1–B4) siguiendo unos documentos de diseño internos que **no forman parte de este
> repositorio**. Las etiquetas se conservan para que el registro del §8 siga siendo trazable contra
> el histórico; cada entrada explica por sí sola qué cambió, por qué y con qué números, sin
> necesidad de consultarlos.

**Medido en**: `2ef8afa` (v3.2.0 + bug fix) · 2026-08-03

---

## 1. Cómo reproducirlo

```bash
python bench/measure.py --notes 2000 --repeat 2      # línea base publicada aquí
python bench/measure.py --notes 200  --repeat 3      # comprobación rápida de varianza
```

El arnés es autocontenido: genera su propio vault, escribe su propio `grimore.toml` en un
directorio de trabajo desechable y hace `chdir` allí. No lee la configuración del usuario, no toca
el vault real y no escribe en el `grimore.db` del repo. Los artefactos generados están en
`.gitignore`; `make_vault.py` reproduce el vault byte a byte desde su semilla.

| Fichero | Qué es |
|---|---|
| `bench/make_vault.py` | Generador determinista de vault (semilla fija) |
| `bench/stub_llm.py` | Servidor determinista con la forma de la API de Ollama |
| `bench/measure.py` | Arnés de medición → `bench/results.json` |

---

## 2. Entorno

| | |
|---|---|
| CPU | Intel Core i7-6500U @ 2.50 GHz (4 hilos) |
| RAM | 15 GB |
| Disco | SSD, ext4 |
| Kernel | 7.0.0-28-generic |
| Python | 3.14.4 (`venv/`) |
| Backend vectorial | **numpy** (fijado con `--vector-backend numpy`) |
| LLM | stub determinista (`bench/stub_llm.py`) |

> **Corrección sobre sqlite-vec.** Las secciones §4–§8 de este documento se midieron **sin**
> sqlite-vec, y afirmaban que la extensión "no carga en este build de Python 3.14". Eso era falso:
> el wheel simplemente no estaba instalado, y `_probe_vec_extension` devuelve `False` ante un
> `ImportError` igual que ante un fallo de carga, así que ambos casos se ven idénticos desde fuera.
> Instalado (`pip install sqlite-vec`), carga sin problemas (v0.1.9) y los 5 tests marcados `vec`
> pasan.
>
> Consecuencia para las mediciones: con la extensión presente, cada embedding hace **dual-write** a
> `embeddings_vec`, así que el scan hace más trabajo que en la línea base del §4. Es otra razón por
> la que los tiempos sólo valen comparados A/B dentro de la misma sesión (§9). La línea base fija
> `--vector-backend numpy` de forma explícita para que el *ranking* use siempre el mismo camino;
> eso no desactiva el espejo, que es de la capa de escritura.

---

## 3. Por qué el LLM está stubbeado (decisión metodológica)

Es tentador medir el `scan` contra un Ollama real. Leyendo el código, no funciona, por dos razones:

1. **`tagger.tag_note()` corre en `cli.py:279`, antes del check de `dry_run`.** Un scan paga una
   llamada LLM por nota incluso en dry-run.
2. **Con el LLM en el bucle, la generación es ~90% del wall-clock.** El criterio de aceptación de la
   opt. 1 era **−30% o más en scan**, pero una mejora del 30% en la capa SQLite es indetectable
   cuando SQLite es el 5% del total. Además la latencia de generación deriva con la residencia del
   modelo y el estado térmico, lo que hace imposible el objetivo de ±5% de reproducibilidad.

El stub responde de forma instantánea y determinista, de modo que el arnés mide **la capa que las
optimizaciones 1–5 tocan de verdad**: parseo, chunking, SQL y numpy.

**Lo que este arnés NO mide, deliberadamente**: calidad de respuesta, calidad de recuperación y
latencia real de extremo a extremo. Eso es competencia de `grimore eval` contra modelos reales. Los
vectores del stub proceden de un hashing vectorizer sobre palabras, así que preservan la
similitud léxica (el ranking hace trabajo real y `connect` encuentra candidatos), pero **no
codifican semántica**. Ninguna afirmación de calidad puede apoyarse en estos números.

---

## 4. Línea base — 2000 notas, semilla 42, 2 corridas

Vault generado: 2000 notas · 17.465 chunks · ~9,4 KB por nota.

### 4.1 Métricas de referencia

| Métrica | Media | Min | Max | Dispersión | ±5% | Cuello / opt. |
|---|---:|---:|---:|---:|:---:|---|
| `scan_s` | **476,09 s** | 474,94 | 477,25 | 0,48 % | ✅ | B1 · opt. 1, 3 |
| `scan_db_opens` | **66.396** | 66.396 | 66.396 | 0,00 % | ✅ | **B1 · opt. 1** |
| `connect_s` | **26,46 s** | 26,37 | 26,54 | 0,63 % | ✅ | B4 · opt. 5 |
| `connect_db_opens` | **12.003** | 12.003 | 12.003 | 0,00 % | ✅ | B4 · opt. 5 |
| `load_dense_s` | **0,1122 s** | 0,1103 | 0,1141 | 3,39 % | ✅ | B3 · opt. 4 |
| `load_dense_peak_mb` | **121,22 MB** | 121,22 | 121,22 | 0,00 % | ✅ | **B3 · opt. 4** |
| `load_dense_rows` | 17.465 | — | — | 0,00 % | ✅ | contexto |
| `load_dense_matrix_mb` | 53,65 MB | — | — | 0,00 % | ✅ | contexto |

### 4.2 `ask` con `top_k=5` — en caliente y en frío

Se miden por separado porque son los **dos llamantes reales**. Un `grimore ask` de un solo uso
construye la matriz densa dentro de su única consulta, así que su `retrieve_s` carga con el coste
completo de `load_dense` y hereda su varianza. El shell mantiene una `Session` viva, de modo que
toda consulta posterior a la primera pega en la caché del connector sellada por firma. El N+1 del
Oracle (B2, opt. 2) vive en el caliente; medirlos juntos no medía ninguno de los dos.

| Métrica | Media | Dispersión | ±5% | Nota |
|---|---:|---:|:---:|---|
| `warm.total_s` | **0,1010 s** | 0,68 % | ✅ | shell / `Session` viva |
| `warm.retrieve_s` | **0,0894 s** | 0,35 % | ✅ | **gate de la opt. 2** |
| `warm.embed_s` | 0,0009 s | 1,03 % | ✅ | caché de embeddings |
| `warm.db_opens` | **13,0** | 0,00 % | ✅ | **gate de la opt. 2** (objetivo: ≤2) |
| `warm.generate_s` | 0,0026 s | 12,81 % | ❌ | 2,6 ms de stub; ninguna opt. lo toca |
| `cold.total_s` | 0,2289 s | 16,59 % | ❌ | CLI de un solo uso |
| `cold.retrieve_s` | 0,2057 s | 18,06 % | ❌ | domina la construcción de matriz |
| `cold.db_opens` | **15,0** | 0,00 % | ✅ | +2 sobre el caliente: carga de matriz |

**Sobre las cuatro métricas que no cumplen el ±5%**: ninguna es un gate. `warm.generate_s` y
`cold.rewrite_s` son etapas de milisegundos o de coste cero (sin historial, `_rewrite_query`
retorna de inmediato) que ninguna optimización ataca. `cold.total_s`/`cold.retrieve_s`
están dominadas por `load_dense`, cuyo coste ya se mide de forma aislada y estable en
`load_dense_s`. Los criterios de aceptación de las opts. 1, 2 y 4 están enunciados en
**conteos** ("de O(chunks) a O(1)", "de ~11 consultas a ≤2"), y los cuatro contadores tienen
**0,00 % de dispersión exacta**.

### 4.3 Comprobación de varianza a 200 notas (n=3)

El baseline publicado usa n=2 por coste de wall-clock (~8 min por corrida). La reproducibilidad
±5% se demostró de forma independiente con n=3 a 200 notas: `scan_s` 4,83 %, `connect_s` 4,67 %,
`warm.retrieve_s` 0,59 %, `warm.total_s` 4,09 %, y todos los contadores 0,00 %. Único fallo a esa
escala: `load_dense_s` (30,8 %), que a 200 notas es una medición de 15 ms donde el jitter del
planificador domina; a la escala de la línea base (2000 notas, 112 ms) baja a 3,39 %.

---

## 5. Lo que la línea base ya demuestra

### B1 — conexión por operación (crítico, opt. 1)

**66.396 aperturas de conexión SQLite para indexar 2000 notas / 17.465 chunks** = **3,80 aperturas
por chunk**. Cada una reconfigura `PRAGMA journal_mode=WAL` y hace su propio `commit`.

El dato demoledor: `476,09 s / 66.396 = 7,17 ms por conexión`. Con un LLM que responde al instante,
el scan está **dominado por el `fsync` del WAL en cada commit**. Esto es lo que hace que las opts. 1
y 3 sean el bloque de mayor apalancamiento de todo el trabajo.

`Session.close()` lo documenta de forma explícita en `session.py:220`:

> *"Database holds no long-lived connection — each call opens one and closes it on exit"*

### B2 — N+1 del Oracle (alto, opt. 2)

**13 conexiones por consulta en caliente** (15 en frío) para un `ask` con `top_k=5`. La estimación
era ~11; el número real es algo peor. Objetivo: ≤2.

### B3 — `_load_dense` arrastra `text_content` (alto, opt. 4)

**Pico de 121,22 MB para una matriz de 53,65 MB** = **67,57 MB (2,26×) de sobrecoste**. Ese
delta es precisamente el `text_content` que `get_all_embeddings_with_id` trae y que la matriz no
necesita — la hipótesis, ahora medida. Y esto con sólo 17.465 chunks; un vault grande proyecta a
100k.

### B4 — `connect` es O(notas × N) (alto, opt. 5)

**26,46 s y 12.003 aperturas de conexión** para 2000 notas = **6,0 aperturas por nota**, encima del
barrido de similitud completo por nota. Confirma la acumulación B1+B2+B4.

---

## 6. Sobre el conteo de conexiones

Lo evidente sería `strace -f -e trace=openat`. El arnés cuenta en su lugar las llamadas a
`sqlite3.connect` dentro del proceso, porque es portable (`strace` necesita permiso de ptrace y no
está en todas partes) y más preciso. **Ambos métodos se validaron cruzados sobre un scan idéntico
en frío de 50 notas**:

| Método | Cuenta |
|---|---:|
| Contador en proceso de `sqlite3.connect` | **1.725** |
| `strace` · `openat` sobre `grimore.db` | **1.725** |
| `strace` · `openat` sobre `grimore.db-wal` | 1.725 |
| `strace` · `openat` sobre `grimore.db-shm` | 1.725 |
| `strace` · total de syscalls | 5.175 |

Coincidencia exacta en el fichero principal, y queda claro por qué contar en la capa de sqlite3 es
más limpio: **cada conexión abre tres ficheros** (principal + WAL + shm), así que el total crudo de
`openat` triplica el número de conexiones.

---

## 7. Definición de "hecho" de la línea base

- [x] `bench/results.json` se genera con un comando y es reproducible dentro del ±5% en todas las
      métricas de gate (los contadores, de forma exacta al 0,00 %).
- [x] Línea base registrada de las 5 métricas: `scan_s`, timings de `ask` desglosados, `connect_s`,
      `load_dense_s` + `peak_mb`, y el conteo de aperturas de conexión.
- [x] Suite completa verde antes de empezar: **907 passed, 11 skipped, 0 failed**;
      `ruff check grimore` limpio.

Los 11 tests saltados lo hacen por dependencias ausentes, no por fallo: 5 `e2e` (el modelo
`nomic-embed-text` no está descargado; el vault usa `nomic-embed-text-v2-moe`), 5 `vec`
(sqlite-vec no estaba instalado entonces) y 1 `reranker` (sin sentence-transformers).

---

## 8. Registro de optimizaciones

### opt. 8 — Parámetros mágicos → configuración ✅

Seis constantes promovidas a claves de `grimore.toml`, todas leídas con
`getattr(..., default)` siguiendo el patrón de la casa:

| Clave | Sección | Default | Constante que sustituye |
|---|---|---:|---|
| `chunk_store_chars` | `[cognition]` | 500 | `reembed.py` `text_truncation` |
| `context_max_chars` | `[cognition]` | 16.000 | `oracle.py` `_ORACLE_CONTEXT_MAX_CHARS` |
| `embed_batch_size` | `[cognition]` | 32 | `embedder.py` `_EMBED_BATCH_SIZE` |
| `circuit_failure_threshold` | `[cognition]` | 5 | `llm_router.py` `_FAILURE_THRESHOLD` |
| `circuit_cooldown_s` | `[cognition]` | 120 | `llm_router.py` `_COOLDOWN_SECONDS` |
| `max_turns` | `[shell]` | 3 | `session.py` `MAX_TURNS` |

**Sin cambio de rendimiento, y esa es la comprobación.** Al ser una exposición pura de
configuración, el criterio de aceptación es que los defaults reproduzcan el comportamiento
anterior. Verificado
ejecutando el arnés a 200 notas después del cambio: **todos los contadores deterministas idénticos
a la línea base** (`scan_db_opens` 6.564, `connect_db_opens` 1.200, `load_dense_rows` 1.721,
`load_dense_matrix_mb` 5,2869, `load_dense_peak_mb` 11,7633, `warm.db_opens` 13, `cold.db_opens`
15). Esto es equivalencia demostrada empíricamente, no sólo por aserción de test.

Detalles de implementación que merecen constar:

- **Patrón de default a nivel de clase.** `Oracle`, `Session` y `LLMRouter` declaran el default
  como atributo de clase y lo sombrean con el valor configurado en `__init__`. Necesario porque
  varios tests construyen estos objetos con `__new__` para aislarlos de la DB y del LLM; sin el
  atributo de clase, el código de producción reventaría con `AttributeError` en esa ruta.
- **`max_turns = 0` desactiva la memoria de verdad.** El corte se comprueba antes de la rebanada:
  `turns[-0:]` es `turns[0:]`, así que plegar el cero dentro del slice habría conservado el
  historial entero en vez de vaciarlo. Cubierto por test.
- **`embed_batch_size` se acota a un mínimo de 1**, porque un paso de cero haría que el bucle
  `range()` de `embed_batch` no avanzara nunca.

Tests: `tests/test_config.py::TestTunableDefaults` (carga y override) y `tests/test_tunables.py`
(15 casos que verifican que los valores **llegan** a su consumidor). La distinción importa: los
tests de carga por sí solos pasarían aunque un consumidor se hubiera quedado leyendo su constante
de módulo. Los cinco cableados se validaron rojo→verde revirtiéndolos uno a uno — ejercicio que
además destapó un hueco: la primera versión del test del embedder sólo comprobaba
`__init__.batch_size` y no fallaba al revertir el consumo en `embed_batch`.

Suite tras el cambio: **925 passed, 11 skipped, 0 failed** · `ruff` limpio.

---

### opt. 1 — Conexión SQLite reutilizable por hilo ✅

Una conexión por hilo, viva mientras viva el `Database`, con los PRAGMAs y la carga de sqlite-vec
aplicados **una sola vez** en lugar de en cada una de las 73 rutas de acceso a datos. Añade
`busy_timeout=5000`, `cache_size=-16000` y `mmap_size=256MB` (inútiles antes: una caché que se
descarta microsegundos después nunca acierta). Cierre explícito vía `Database.close()`, llamado
desde `Session.close()` y `daemon.stop()`.

| Métrica | pre-opt.1 | opt.1 | Cambio | Criterio |
|---|---:|---:|---:|---|
| `scan_db_opens` | 66.396 | **1** | −100 % | O(1) por hilo ✅ |
| `connect_db_opens` | 12.003 | **1** | −100 % | ✅ |
| `scan_s` | 529,75 s | **65,14 s** | **−87,7 %** | −30 % o más ✅ |
| `connect_s` | 50,89 s | **31,72 s** | **−37,7 %** | (no exigido) ✅ |

Medido A/B en la misma sesión, 2000 notas, 2 corridas cada variante. Ver §9 sobre por qué los
tiempos se miden así y no contra la tabla del §4.

**Diseño.** Tres piezas que merecen constar:

- **Registro de conexiones + contador de generación.** `threading.local` no permite que un hilo
  limpie el slot de otro, pero el daemon indexa en el hilo del observador mientras el principal
  lee, así que `close()` tiene que alcanzar ambas. El registro las reúne; el contador de generación
  invalida los handles cacheados sin tocar el almacenamiento ajeno, y cada hilo lo detecta y
  reabre en su siguiente llamada. Eso hace que reutilizar el `Database` tras un `close()` funcione.
- **`check_same_thread=False` es seguro aquí** sólo porque `_thread_conn` garantiza exactamente una
  conexión por hilo. El flag existe para que `close()` pueda reclamar conexiones ajenas en el
  apagado, no para compartirlas.
- **VACUUM no se ve afectado**: `upkeep.py:30` ya abría su propia conexión dedicada con
  `isolation_level=None`, porque VACUUM se niega a correr dentro de una transacción.

**La verificación previa que hacía falta.** El riesgo real de compartir conexión no es el
rendimiento sino la **re-entrada**: si un método abre una conexión y dentro llama a otro que
también la abre, el `with conn:` interno haría commit de la transacción externa antes de tiempo,
convirtiendo un rollback-por-error en una escritura parcial. Comprobado por dos vías
independientes antes de tocar nada — una sonda en ejecución sobre los 925 tests más el arnés, y un
análisis AST estático — con **0 anidamientos sobre 73 métodos** que abren conexión. El invariante
lo protege ahora `TestNoReentrancy`.

**Verificación funcional con Ollama real** (no el stub), sobre un vault de prueba:

| Superficie | Resultado |
|---|---|
| `preflight`, `scan`, `status`, `ask`, `connect`, `tags`, `dedupe` | OK; `ask` responde con la cita correcta |
| Idempotencia | Segundo scan: 0 procesadas, 2 sin cambios (fast-skip por doble hash) |
| Frontmatter | Escrito correctamente en la nota fuente |
| HTTP API | 200 peticiones: descriptores 8 → 12 y estables; sin errores |
| Daemon | Indexa y reindexa; descriptores 11 → 15 → 15; parada limpia con SIGTERM |
| MCP | Handshake, `tools/list` y `tools/call` correctos |
| **Daemon + CLI concurrentes** | 50 lecturas de CLI con el daemon indexando: **0 errores**, `integrity_check` ok, 0 embeddings huérfanos |

**Gate de concurrencia (innegociable).** `tests/test_connection_reuse.py`, 16 casos: una conexión
por hilo y reutilizada, conexiones distintas entre hilos, `close()` libera de verdad (más conteo de
descriptores en `/proc/self/fd`, regresión inversa del fix de v3.2.0), reutilización tras cierre,
idempotencia, barrido de hilos muertos sin tocar los vivos, y el gate propiamente dicho — un escritor con tres lectores y tres escritores
concurrentes sin `database is locked`, sin filas perdidas y con `PRAGMA integrity_check` en `ok`.
Validado rojo→verde: con `busy_timeout=0` el test de escrituras concurrentes falla; con 5 s, pasa.

**Fuga de descriptores en hilos efímeros (encontrada y corregida en la verificación).** La primera
versión guardaba las conexiones en una lista. El almacenamiento thread-local muere con el hilo,
pero la lista mantenía una referencia fuerte, así que la conexión y su descriptor sobrevivían hasta
`close()`. Reproducido: 100 hilos efímeros dejaban **101 conexiones y 207 descriptores abiertos**,
con crecimiento lineal; con el límite habitual de 1024 FDs, un daemon o una API de larga vida
terminaría en `EMFILE`. Alcanzable en producción vía `ThreadPoolExecutor` — que es el modelo del
threadpool de anyio sobre el que Starlette ejecuta los handlers síncronos, y el que introduciría la
opt. 7.

El arreglo: el registro pasa a ser un diccionario indexado por hilo, y al abrir una conexión nueva
se barren y cierran las de hilos ya muertos. Barrer ahí y no en cada llamada lo mantiene fuera del
camino caliente: esa rama corre una vez por hilo, no una vez por consulta. La vivacidad se comprueba
contra el objeto `Thread` y no contra el ident, porque el sistema operativo recicla idents y un hilo
nuevo podría heredar el de uno muerto. Tras el arreglo las conexiones quedan acotadas (2–9 en las
mismas pruebas) y los descriptores planos.

Suite tras el cambio: **941 passed, 11 skipped, 0 failed** · `ruff` limpio · `mypy` sin issues.

---

### opt. 3 — Insert de embeddings por lote ✅

`store_embeddings_bulk` con `executemany` y **una transacción por nota** en lugar de una por chunk.
`reembed_note` acumula las filas y llama una vez; `store_embedding` (singular) se conserva intacto
como camino de rollback y porque lo usan una veintena de tests.

Medido A/B en la misma sesión, 600 notas, con sqlite-vec instalado (dual-write activo):

| Métrica | por chunk | por lote | Cambio |
|---|---:|---:|---:|
| `scan_s` | 16,234 s | **13,610 s** | **−16,2 %** |
| `scan_db_queries` | 65.375 | **56.839** | **−13,1 %** |

Encima del −87,7 % que ya aportó la opt. 1. El ahorro real es el `fsync`: en WAL cada commit lo
paga, y el bucle antiguo hacía uno por chunk. `executemany` sólo quita, además, la sobrecarga de
Python por fila.

**Por qué los rowids se releen en vez de calcularse.** El atajo evidente es
`first_id = cur.lastrowid - len(rows) + 1`. Ese código **no funciona**: desde Python 3.11
`cursor.lastrowid` queda en `None` tras un `executemany`, así que la resta lanza `TypeError`.
Verificado, y los tests de paridad lo rechazan (3 fallos al sustituirlo). Una variante con
`MAX(id)` sí sería correcta *dentro* de la transacción —WAL admite un solo escritor, así que nadie
puede intercalar una fila mientras se sostiene el bloqueo—, pero el read-back no necesita ese
razonamiento para ser correcto y no se rompe si mañana cambia el modelo de bloqueo.

El read-back selecciona por `note_id` en lugar de un `IN (chunk_index, ...)`: un PDF grande puede
superar el límite de parámetros de SQLite, y las filas de más (los chunks conservados, no
re-embebidos) simplemente no se consultan.

**Gate de paridad del espejo vec.** `tests/test_bulk_embeddings.py`, 14 casos, de los cuales 6
ejercitan `embeddings_vec` con la extensión real: conteos de fila iguales, **cada vector archivado
bajo su propio rowid** (el fallo que un mapeo desplazado produce es una cita apuntando a otra nota,
silencioso y sólo detectable comparando el vector con su fila origen), supervivencia a un
re-embed incremental, encogimiento al acortar la nota, ausencia de contaminación entre notas, y
mismatch de dimensión que igual persiste la fila fuente. Más un test de que diez chunks producen
**un solo COMMIT**.

**Bug preexistente corregido durante la verificación: el espejo vec podía quedar roto en silencio.**
`_create_vec_table` fija `self._vec_dim` en Python *dentro* de la transacción que crea
`embeddings_vec`. Si esa transacción revierte después, el DDL se deshace pero el atributo
sobrevive, así que toda escritura posterior se salta la creación e inserta en una tabla que ya no
existe. El `except OperationalError` lo degradaba a un `warning` y el scan continuaba: los
`embeddings` seguían guardándose y **ningún vector se reflejaba**, dejando un índice vec que
responde con una fracción del vault. Preexistente y no introducido por la opt. 3 — reproducido
igual contra `store_embedding`, el camino de una fila que este trabajo no toca.

El arreglo recupera **dentro de la misma llamada** en vez de en la siguiente: las filas de
`embeddings` del lote que falla se comprometen igualmente, así que nadie volvería nunca a
reflejarlas. Al fallar la escritura se limpia `_vec_dim`, se recrea la tabla (`CREATE ... IF NOT
EXISTS`, así que el reintento es seguro aunque el fallo fuera otro) y se reintenta una vez. Sigue
la convención que ya usaba `drop_vec_table`, que resetea el dim tras un DROP explícito; al camino
de rollback simplemente le faltaba el equivalente. Cubierto por 3 tests, validados rojo→verde.

Suite tras el cambio: **963 passed, 6 skipped, 0 failed** · `ruff` limpio · `mypy` sin issues.
Sin sqlite-vec instalado (configuración por defecto de CI): 949 passed, 20 skipped, 0 failed.

### opt. 2 — Matar el N+1 del Oracle ✅

La propuesta mínima, extendida también a las anclas: `get_note_titles(ids)` y
`get_chunk_anchors_bulk(pairs)`. El retrieval no se toca — ni `JOIN`, ni cambio en la semántica de
match—, así que empujar los metadatos a la propia consulta de recuperación queda disponible si un
bench futuro lo justifica.

**Por qué no se usó el `embedding_id`.** Los dos caminos de retrieval ya lo conocen y lo descartan
(`connector.py`, "Drop the internal embedding_id"), y hay un test que fija ese contrato. Con él,
las anclas serían una búsqueda por clave primaria en vez de un match por `text_content`. Es la vía
más rápida y estrictamente más correcta —dos chunks de una nota con los mismos 500 caracteres
almacenados hoy pueden devolver el ancla del otro—, pero cambia la semántica observable y rompe un
contrato con test. Es el paso natural siguiente.

#### Gate 1 — presupuesto de consultas (independiente de la sesión)

Vault de 200 notas, `--count-queries`, A/B en la misma sesión:

| Métrica | antes | después | Cambio |
|---|---:|---:|---:|
| `ask` (caliente) sentencias de aplicación | 13 | **5** | **−8** |
| └ de las cuales, metadatos | 10 | **2** | **−80 %** |
| `ask_cold` sentencias de aplicación | 17 | **9** | **−8** |

10 → 2 es exactamente el criterio de aceptación ("de ~11 a ≤2"). Las 3 sentencias restantes en
caliente son control transaccional, no metadatos.

#### Gate 2 — paridad de citas

`_build_context` sobre el mismo vault y la misma pregunta, antes y después: **`context` idéntico
byte a byte** (2814 chars) y `retrieved` idéntico (mismos note_id, mismo rank, mismo score).

Durante esta comprobación apareció una diferencia en `sources`: **mismo conjunto, orden distinto**.
No es una regresión de la opt. 2 — `_build_context` devuelve `list(set(sources))` (`oracle.py:472`),
y el orden de iteración de un `set` de strings depende de `PYTHONHASHSEED`, que se aleatoriza por
proceso. Verificado ejecutando el **mismo** código en tres procesos: tres órdenes distintos, mismo
contexto. Es no-determinismo preexistente; el propio código lo documenta ("``sources`` is flattened
through ``set()`` and can't carry order"). Queda anotado como wart, no como bloqueo.

#### Gate 3 — tiempo

15 corridas por brazo, 5 rondas alternadas `antes/después` para promediar la deriva térmica
(§9), 200 notas:

| `ask` caliente | antes (mediana) | después (mediana) | Cambio | IQRs |
|---|---:|---:|---:|---|
| hueco sin instrumentar | 2,251 ms | **1,340 ms** | **−40,5 %** | apenas se solapan |
| `retrieve_s` | 11,194 ms | 11,336 ms | +1,3 % | — |
| `embed_s` (control) | 0,119 ms | 0,119 ms | +0,2 % | — |
| `generate_s` | 3,556 ms | 2,967 ms | −16,6 % | **se solapan** |
| `total_s` | 17,946 ms | 15,892 ms | −11,4 % | — |

**El criterio enunciado apuntaba al bucket equivocado.** Pedía que bajase `retrieve_s`, pero el
Oracle cierra ese cronómetro (`oracle.py:380`) *antes* del bucle de metadatos. El trabajo que
elimina la opt. 2 cae en el hueco entre `retrieve_s`/`rerank_s` y `generate_s`, que ninguna etapa
cubre; de ahí la fila "hueco" (`total_s` menos la suma de las etapas). Ahí el efecto es
inequívoco: −40,5 %, ~0,9 ms por `ask`.

`generate_s` bajando un 16,6 % **no es atribuible a este cambio**: el stub es determinista y el
contexto es idéntico byte a byte, así que el payload es el mismo. Con 15 muestras las IQRs se
solapan de sobra — es ruido. `embed_s`, que sirve de control, queda plano al +0,2 %, lo que
confirma que el arnés no derivó durante la tanda.

El ahorro absoluto (~0,9 ms) es pequeño porque el vault es chico y la BD está caliente; escala con
el tamaño del pool recuperado y pesa mucho más en almacenamiento lento.

Suite tras el cambio: **981 passed, 6 skipped, 0 failed** · `ruff` limpio · `mypy` sin issues en 73
ficheros. `tests/test_batch_metadata.py` añade 13 casos, con las tres mutaciones clave
(`ORDER BY id` invertido, filtro de pares eliminado, dedup eliminado) verificadas rojo→verde.

---

## 9. Limitación del arnés: los tiempos no son comparables entre sesiones

Descubierto al verificar la opt. 1, y afecta a **todas** las optimizaciones que quedan.

Comparar `connect_s` tras la opt. 1 contra la línea base del §4 daba **+19,9 %**, una regresión
aparente. Medir las dos variantes seguidas en la misma sesión da **−37,7 %**. La diferencia no
está en el código: el mismo código pre-opt.1 rinde `connect_s` = 26,46 s en la sesión de la línea
base y **50,89 s** en la sesión del experimento A/B — casi el doble.

La causa es la máquina: un i7-6500U es un chip móvil de 2 núcleos que estrangula por temperatura
bajo carga sostenida. Y el propio éxito de la opt. 1 cambia el perfil térmico del workload: antes
el scan pasaba 476 s esperando `fsync` con la CPU fría, ahora hace el mismo trabajo en 65 s de CPU
densa, así que `connect` arranca sobre un procesador ya caliente.

**Regla para las siguientes optimizaciones:**

- Los **contadores** (`*_db_opens`, `*_db_queries`, `load_dense_peak_mb`, `*_rows`) son
  deterministas y sí comparables entre sesiones. Las tablas del §4 siguen siendo válidas para ellos.
- Los **tiempos** (`scan_s`, `connect_s`, `*_total_s`) sólo son válidos comparados **A/B dentro de
  la misma sesión**: se guarda el fichero anterior, se mide, se aplica el cambio, se mide otra vez.
  La tabla del §4 sirve de orden de magnitud, no de referencia para un delta.

### Otras dos cautelas del arnés

- **El trazado de sentencias distorsiona los tiempos.** El callback cuesta ~1,2 µs por sentencia y
  un `ask` ejecuta >17.000, o sea ~20 ms sobre una medición de 100 ms — suficiente para reportar la
  instrumentación como regresión. Por eso `--count-queries` es opt-in: las corridas de conteo y las
  de cronometraje son pasadas separadas, igual que `tracemalloc` en `bench_load_dense`.
- **El vault sintético tiene un vocabulario de 50 palabras**, así que toda nota contiene casi todo
  el léxico y una consulta FTS casa contra *todos* los chunks. Eso infla las sentencias internas de
  FTS5 (`bm25()` lee una fila de `docsize` por documento que casa: 1.535 de las 1.552 sentencias de
  un `ask` son internas; las de Grimore son ~17). No invalida ninguna comparación antes/después,
  porque el vault es constante, pero hay que tenerlo presente en la **opt. 9**, donde la
  selectividad del filtro es justamente el objeto de la medida.

---

## 10. Siguiente paso

Cerradas las opts. 8 → 1 → 3 → 2, el siguiente bloque es una cadena independiente:

1. **opt. 4** — `_load_dense` sin `text_content` + caché `.npy` con `mmap` (2 d, riesgo bajo).
   Gate: `load_dense_peak_mb` baja, y la caché se invalida al cambiar el vault.
2. **opt. 5** — `connect` vectorizado, `M @ M.T` por bloques (1,5 d, riesgo bajo). Depende de la
   opt. 4 y reutiliza el `get_note_titles(ids)` que acaba de introducir la opt. 2. Gate: los
   enlaces sugeridos por encima del umbral son **los mismos**.

La línea base para ambos ya está tomada: `load_dense_peak_mb` y `connect_s` en §4.1.

Recordatorio al medir: el vault sintético tiene un vocabulario de 50 palabras, así que la mayor
parte de las sentencias de un `ask` son internas de FTS5, no de Grimore. El contador ya las separa
(`db_queries` frente a `db_queries_internal`); el gate es el primero.

Y una lección del gate 3 de la opt. 2, aplicable a todo lo que venga: **incluir siempre una métrica
de control** que el cambio no pueda tocar (`embed_s` sirvió aquí). Sin ella no hay forma de
distinguir una ganancia real del ruido de la máquina, y con 2–3 corridas es fácil convencerse de
que un movimiento del 17 % es señal cuando no lo es.
