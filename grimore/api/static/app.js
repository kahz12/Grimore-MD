// =================================================================
// Grimore — Codex front-end controller.
// Vanilla JS. Streams answers via SSE, parses [[citation]] markers
// into clickable glyphs and opens notes in a side drawer. It runs directly in
// the browser without bundling or runtime compilation.
// =================================================================

const $  = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
    health: null,
    categories: [],
    busy: false,
    lastSearchHits: [],
    setup: null,
    indexing: null,
    reviewInbox: null,
    reviewFilter: 'all',
    filterOptions: null,
    conversation: [],
    sourceDetails: [],
    citationSelections: null,
    sourceLabels: [],
    retrievalScope: null,
    evidenceStatus: null,
    writeEnabled: false,
    writeToken: '',
    operationBusy: false,
    reviewDecisionBusy: false,
    reviewClassificationBusy: false,
    synthesisBusy: false,
    synthesisPreview: null,
    restartRequired: false,
};

const CONVERSATION_CONTEXT_TURNS = 3;

const I18N = {
    en: {
        pageTitle: 'Grimore — An Arcane Knowledge Engine', language: 'Language',
        setup: 'Setup', setupEyebrow: 'First-run assistant · read-only', closeSetupLabel: 'Close setup',
        setupTitle: 'Prepare your library', closeSetup: 'Close ✕',
        setupIntro: 'Validate the running configuration, choose from reported models, and copy or securely apply the resulting TOML values.',
        setupInitial: 'Open the assistant to validate this installation.',
        libraryFolder: 'Library folder',
        libraryFolderHelp: 'Only the currently running folder is checked on the server. Edit this draft, copy it, then restart Grimore.',
        chatModel: 'Chat model', embeddingModel: 'Embedding model', validation: 'Validation',
        scanEffects: 'What a scan can change', valuesFor: 'Values for',
        configHelp: 'Replace the matching keys in the existing sections; avoid creating duplicate section headers.',
        recheckConfig: 'Recheck running config', copyToml: 'Copy TOML values',
        activityEyebrow: 'Library activity · read-only', indexingTitle: 'Indexing ledger',
        checking: 'Checking…', lookingForScan: 'Looking for a recent scan…', noIndexingEvents: 'No indexing events recorded yet.',
        indexingProgress: 'Indexing progress', indexingResults: 'Indexing results',
        processed: 'Processed', skipped: 'Skipped', failed: 'Failed', previewed: 'Previewed', removed: 'Removed', chunks: 'Chunks',
        recentFiles: 'Recent files', recovery: 'Recovery',
        indexingNote: 'Protected actions require the API token. Preview does not change documents; apply uses the normal scan safety controls.',
        changeHistory: 'Applied document changes',
        changeHistoryIntro: 'Private pre-change copies can restore files only while their post-write version is still current.',
        changeRetention: 'Latest 20 change sets retained',
        changeHistoryEmpty: 'No recoverable document changes recorded.',
        changeFiles: '{count} changed file{suffix}', restoreFiles: 'Restore files',
        restoreConfirm: 'Restore the copied versions from before this operation? Later edits cause a safe refusal.',
        restoringFiles: 'Restoring copied files…', restoreComplete: 'Files restored. Run a preview scan before applying indexing again.',
        restoreFailed: 'The files could not be restored: {detail}',
        restoreConflict: 'Restoration was refused because one or more files changed afterwards.',
        previewChanges: 'Inspect proposed changes', previewCreate: 'Create', previewUpdate: 'Update',
        previewTarget: 'Target: {value}', beforeValue: 'Before', afterValue: 'After', emptyValue: '— empty —',
        reviewEyebrow: 'Maintenance · candidates only', reviewInbox: 'Review inbox', show: 'Show',
        reviewAll: 'All candidates', contradictions: 'Contradictions', exactDuplicates: 'Exact duplicates',
        nearDuplicates: 'Near duplicates', reviewDue: 'Review due', refreshInbox: 'Refresh inbox',
        reviewCandidateNotice: 'These are candidates for human review, not confirmed errors or automatic decisions.',
        reviewCounts: 'Review candidate counts', reviewLoading: 'Loading maintenance signals…',
        reviewUnavailable: 'Review inbox unavailable: {detail}', reviewEmpty: '— no candidates in this view —',
        reviewShowing: '{count} candidate{suffix} shown.', candidate: 'Candidate', documents: 'Documents',
        synthesisEyebrow: 'Knowledge draft · no write', topicSynthesis: 'Topic synthesis',
        synthesisNeedsReview: 'Needs human review',
        synthesisIntro: 'Generate a temporary draft from indexed passages. It is not saved to the library.',
        synthesisKind: 'Group by', synthesisTag: 'Tag', synthesisCategory: 'Category',
        synthesisValue: 'Topic value', synthesisPlaceholder: 'e.g. history',
        synthesisPassagesPerNote: 'Passages per note', generateSynthesis: 'Generate preview',
        synthesisIdle: 'Choose a tag or category to inspect a synthesis draft.',
        synthesisLoading: 'Building a reviewable draft…',
        synthesisFailed: 'The synthesis preview could not be generated: {detail}',
        synthesisNoOutput: 'No draft was produced: {detail}', synthesisNoReason: 'no usable output',
        synthesisReady: 'Draft ready for review. Inspect its coverage and source passages before relying on it.',
        synthesisPreviewOnly: 'Preview only · not written', synthesisSources: 'Inspectable sources',
        synthesisNoSources: 'No indexed source passages are available to inspect.',
        synthesisCoverageComplete: 'Complete for the configured scope',
        synthesisCoveragePartial: 'Incomplete coverage', synthesisCoverageUnavailable: 'Coverage unavailable',
        synthesisNotesMetric: '{included} of {eligible} eligible notes included',
        synthesisPassagesMetric: '{selected} of {available} indexed passages selected',
        synthesisContextMetric: '{used} of {limit} context characters used',
        synthesisSourceMetric: '{selected} of {available} passages selected',
        synthesisSourceIncluded: 'Included in draft', synthesisSourceOmitted: 'Omitted by context limit',
        synthesisSourceUnavailable: 'No indexed passages',
        synthesisOpenSource: 'Inspect source document',
        synthesisLimitGenerated: 'Previously generated synthesis notes were excluded to prevent feedback loops.',
        synthesisLimitMissingPassages: 'Some matching notes have no indexed passages and could not be used.',
        synthesisLimitContext: 'The context limit omitted one or more otherwise eligible notes.',
        synthesisLimitRepresentative: 'Only representative passages were selected; the full indexed text was not sent to the model.',
        synthesisLimitUnknownCitations: 'The draft contains citations that do not match an included source.',
        synthesisLimitUncitedSources: 'One or more included sources are not cited in the draft.',
        synthesisLimitNoMatches: 'No indexed notes match this topic.',
        exactDuplicateReason: '{count} documents share the same indexed content.',
        nearDuplicateReason: 'Semantic similarity {score} (threshold {threshold}).',
        reviewDueReason: 'Review window exceeded by {days} day{suffix}.',
        contradictionFallback: 'The indexed claims may conflict.',
        inspectCandidate: 'Inspect evidence and coverage', reasonLabel: 'Why it was flagged',
        analysisCoverage: 'Analysis coverage', involvedDocuments: 'Involved documents',
        candidatePassages: 'Passages for review', openDocument: 'Inspect document',
        noCandidatePassages: 'No indexed passage is available for this candidate.',
        coverageComplete: 'Complete for this signal', coveragePartial: 'Partial', coverageUnavailable: 'Unavailable',
        coverageDocuments: '{analyzed} of {total} documents analyzed',
        coveragePassages: '{shown} of {available} review passages shown',
        coverageChunks: '{count} chunks compared',
        coverageClaimCap: 'Claim extraction analyzes up to {sections} source parts ({characters} characters total) and returns up to {claims} claims per document.',
        coverageResultCap: 'At most {count} similar pairs are included in one inbox response.',
        basisClaimPair: 'One extracted claim pair', basisContentHash: 'Normalized full-text hash',
        basisMeanEmbeddings: 'Mean of indexed chunk embeddings', basisReviewWindow: 'Configured category review window',
        limitClaimPair: 'Only the flagged claim pair was evaluated, not every statement in both documents.',
        limitClaimExtractionBounded: 'Claim extraction is intentionally bounded for long documents.',
        limitModelCandidate: 'A model proposed this conflict; human confirmation is required.',
        limitRepresentativePassages: 'Displayed passages are representative; the signal may cover more indexed text.',
        limitMetadataIgnored: 'Document metadata does not participate in the exact-content comparison.',
        limitSemanticNotIdentity: 'Semantic proximity does not prove duplication or factual equivalence.',
        limitResultLimit: 'The inbox shows a bounded number of the strongest similar pairs.',
        limitAgeNotValidity: 'Elapsed time does not prove that the content is outdated.',
        extractedClaim: 'Extracted claim', indexedPassageLabel: 'Indexed passage',
        characterRange: 'Characters {start}–{end}',
        reviewActions: 'Review decision', confirmCandidate: 'Confirm finding', dismissCandidate: 'Dismiss',
        postponeCandidate: 'Postpone', postponeFor: 'Postpone for', days7: '7 days', days30: '30 days', days90: '90 days',
        reviewAuthHelp: 'Decisions require the protected operations token from Setup.',
        reviewDecisionSaving: 'Saving review decision…', reviewDecisionSaved: 'Decision saved in the audit history.',
        reviewDecisionFailed: 'The review decision could not be saved: {detail}',
        classificationTitle: 'Document classification', classificationCategory: 'Category',
        classificationTags: 'Tags', classificationTagsHelp: 'Comma-separated tags',
        classificationManual: 'Manual correction active', classificationIndexed: 'Current indexed classification',
        classificationSave: 'Save classification',
        classificationScope: 'Updates the index and preserves this choice for future scans; it does not edit the source document now.',
        classificationSaving: 'Saving category and tags…',
        classificationSaved: 'Manual classification saved; future scans will preserve it.',
        classificationFailed: 'The classification could not be saved: {detail}',
        documentDates: 'Document dates', datePublished: 'Published', dateModified: 'Modified',
        dateIndexed: 'Indexed', dateVerified: 'Last verified', dateUnknown: 'Unknown',
        dateSource: 'source: {value}',
        dateSourceDocument: 'document metadata ({field})', dateSourceFilesystem: 'file modification time',
        dateSourceIndex: 'Grimore indexing', dateSourceHuman: 'human verification',
        dateSourceLegacy: 'legacy record (unspecified)',
        decisionHistory: 'Decision history', historyEmpty: '— no review decisions recorded —',
        decisionConfirm: 'Confirmed', decisionDismiss: 'Dismissed', decisionPostpone: 'Postponed',
        decisionRecordedAt: 'Recorded {value}', postponedUntil: 'until {value}',
        catalogue: 'Catalogue', catalogueHelp: 'Press a heading to ask after its contents.',
        conversationTitle: 'Conversation memory', newConversation: 'New conversation',
        invocation: 'Invocation', questionPlaceholder: 'Pose a question to the codex…',
        searchScope: 'Search scope', category: 'Category', categoryHelp: 'Includes descendant categories.',
        tags: 'Tags', tagsHelp: 'Comma-separated; every tag must match.', formats: 'Formats',
        formatsHelp: 'Selecting several formats includes any of them.', clearScope: 'Clear scope',
        passages: 'passages', invoke: 'Invoke', toInvoke: 'to invoke', loadingFormats: 'Loading formats…', revelation: 'Revelation',
        citedSources: 'Cited Sources', relatedPassages: 'Related Passages',
        marginalHelp: 'Marginal annotations appear here when you invoke a question.',
        footerCreed: 'read-only · offline-capable · models',
        allDocuments: 'All indexed documents', allCategories: 'All categories',
        noConversation: 'No prior turns · the next question starts a new conversation.',
        conversationHelp: 'The three most recent answers will provide context for follow-up questions.',
        contextTurns: '{count} of {max} context turns retained.', citedCount: '{count} cited source{suffix}',
        scopeCategory: 'category: {value}', scopeTags: 'tags: {value}', scopeFormats: 'formats: {value}',
        scopeApplied: 'This scope applies to both the answer and related passages.',
        noPassagesYet: '— no passages summoned yet —', noPassages: '— no passages summoned for this question —',
        noScopeMatches: '— no indexed documents match the selected scope —',
        requestWorking: 'Inscribing…', requestFailed: 'The request failed: {detail}',
        searchFailed: 'Related passages could not be loaded: {detail}', networkError: 'The server could not be reached.',
        noTokens: '— the codex returned no tokens. Is the model awake? —',
        answerFailed: '— the codex falls silent. The backend may be unreachable. —',
        inspectEvidence: 'Inspect indexed passage and source state',
        evidenceSupported: 'Evidence available · inspect {count} indexed passage{suffix}.',
        evidenceUnavailable: 'Evidence was cited, but inspectable passage details are unavailable.',
        evidenceConflict: 'Conflicting evidence · compare the {count} retrieved passages before relying on the answer.',
        evidenceUncertain: 'Uncertain answer · inspect the retrieved passages and treat the conclusion cautiously.',
        evidenceInsufficient: 'Insufficient evidence · Grimore did not retrieve enough support for this answer.',
        evidenceNoMatches: 'Insufficient evidence · no indexed documents match the selected scope.',
        evidenceError: 'Evidence status unavailable · the request did not complete.',
        evidencePdfPartial: 'Incomplete PDF coverage ({count}) · some pages in cited documents had no extractable text when indexed. The answer may omit information.',
        evidencePdfUnknown: 'Unverified PDF coverage ({count}) · extraction status was not recorded when indexed. Re-scan before relying on whole-document claims.',
        setupDraft: 'Draft changed · copy these values and restart Grimore before rechecking.',
        currentSelection: 'currently selected', setupReady: 'Ready for a preview scan · no changes have been made.',
        setupAttention: 'Setup needs attention · review the checks below.', setupChecking: 'Checking folder and model services…',
        setupFailed: 'Setup validation failed: {detail}', copied: 'Copied', copyUnavailable: 'Copy unavailable',
        preview: 'Preview', apply: 'Apply', protected: 'Protected',
        vaultCheckPass: 'Configured folder is readable.', vaultCheckError: 'Configured folder is missing, unreadable, or not a directory.',
        vaultCheckFix: 'Update [vault].path, restart Grimore, and validate again.',
        chatCheckPass: 'Chat model “{model}” is available.', chatCheckError: 'Chat model “{model}” was not reported by the backend.',
        chatCheckFix: 'Select a reported chat model or install it in the configured backend.',
        embeddingCheckPass: 'Embedding model “{model}” is available.', embeddingCheckError: 'Embedding model “{model}” was not reported by Ollama.',
        embeddingCheckFix: 'Select a reported embedding model or pull it with Ollama.',
        configCheckWarning: 'No grimore.toml exists in the server working directory.', configCheckFix: 'Create grimore.toml there, then restart Grimore.',
        scanPreviewRead: 'Reads changed documents and sends their extracted text to the chat model.',
        scanPreviewShow: 'Shows proposed tags, summary, and category.', scanPreviewNoWrite: 'Does not rewrite documents, sidecars, or indexed note records.',
        scanApplyMarkdown: 'Writes generated frontmatter into changed Markdown files.', scanApplySidecars: 'Creates or refreshes managed sidecars for non-Markdown files when enabled.',
        scanApplyIndex: 'Stores note metadata, tags, categories, chunks, and embeddings in SQLite.', scanApplyGit: 'Creates a Git safety snapshot before document changes when enabled and available.',
        scanNeverOriginals: 'Never modifies the original PDF, EPUB, DOCX, ODT, RTF, HTML, or TXT file.', scanNeverPrivate: 'Skips documents marked privacy: never_process.',
        writeToken: 'Protected operations token', writeTokenHelp: 'Kept only in this tab’s memory and sent solely to protected write operations.',
        saveConfig: 'Save configuration', writeDisabled: 'Protected writes are disabled. Restart grimore serve with an API token.',
        tokenRequired: 'Enter the protected operations token in Setup.', configSaving: 'Saving the scoped configuration…',
        applyConfigConfirm: 'Save only the library path and selected models to grimore.toml?',
        configSavedRestart: 'Configuration saved. Restart Grimore to use the new values.', configUnchanged: 'Configuration already matches these values.',
        previewIndexing: 'Start safe preview', applyIndexing: 'Apply indexing', applyIndexingConfirm: 'Apply indexing now? This may update Markdown metadata, sidecars, the index, and Git snapshots.',
        indexingStarting: 'Starting indexing…', indexingAccepted: 'Indexing accepted; progress will appear here.',
        indexingFinished: 'Indexing process finished.', indexingExitFailed: 'Indexing process exited with code {code}. Review the recovery details.',
        restartBeforeIndexing: 'Restart Grimore before indexing with the saved configuration.',
        operationFailed: 'Protected operation failed: {detail}', invalidToken: 'The protected operations token is invalid.', indexingActive: 'An indexing operation is already running.',
        indexingRunning: 'Indexing now', indexingComplete: 'Complete', indexingErrors: 'Complete with errors',
        indexingInterrupted: 'Interrupted', indexingStopped: 'Watcher stopped', noRun: 'No run recorded',
        previewFirst: 'Run grimore scan --dry-run to preview the first indexing pass.',
        nowReading: 'Now reading: {path}', filesExamined: '{completed} of {total} files examined.',
        watcherReady: 'Watcher is ready for file changes.', statusUnavailable: 'Status unavailable',
        indexingUnavailable: 'Could not read indexing activity: {detail}', noFileResults: 'No file results recorded for this library yet.',
        unknownFile: 'Unknown file', noRecovery: 'No recovery action is needed.',
        noFormats: 'No formats are configured.', scopeUnavailable: 'Scope options unavailable: {detail}', formatsUnavailable: 'Formats unavailable.',
        catalogueUnavailable: '— catalogue unavailable —', noCategories: '— no categories indexed —', uncategorised: '— uncategorised —',
        categoryQuestion: 'What does the codex contain on {category}?', marginalGloss: 'Marginal Gloss', note: 'note', score: 'score',
        enterQuestion: 'Enter a question before invoking the codex.', noMatchesHelp: 'No indexed documents match this scope. Clear or adjust the filters to search elsewhere.',
        citationsRemoved: '{count} citation{suffix} produced by the model were not in the retrieved context and were removed.',
        close: 'Close', sourceEvidence: 'Source Evidence', retrievedPassages: 'Retrieved passages', previous: 'Previous', next: 'Next',
        chooseCitationPassage: 'Choose a passage',
        ambiguousCitation: 'This reference matches several retrieved passages. Choose one to inspect; the citation does not identify a single passage.',
        unverifiableSelection: 'The selected passage cannot be verified from this response. These are matching retrieved passages, not a confirmed model selection.',
        selectedCitation: 'Inspect the passage selected by this citation',
        chooseCitation: 'Choose among matching passages',
        indexedPassage: 'Indexed passage', currentDocument: 'Current document · passage highlighted', goTo: 'Go to page or section',
        locatingPassage: '— locating the passage —', source: 'Source', noDocumentPath: 'No current document path',
        noIndexedPassage: 'No indexed passage is available.', page: 'Page {value}', section: 'Section: {value}', passage: 'Passage {value}',
        excerptOnly: 'Excerpt only', noLocator: 'No page or section locator available', position: '{current} of {total}',
        passageNotLocated: 'The exact passage could not be located in the current rendering; use the indexed passage above.',
        extractionPartial: 'Extraction is incomplete. No text was recovered from page(s) {pages}; answers may miss information there.',
        extractionRecovered: 'OCR or a fallback recovered text from page(s) {pages} in the current PDF. Check those passages against the original; indexed answers may need refreshing.',
        extractionFailed: 'The current document could not be read. The indexed passage above may be outdated or incomplete.',
        extractionOcrUnavailable: 'OCR tools are unavailable.',
        extractionOcrRenderFailed: 'The page could not be rendered for OCR.',
        extractionOcrRecognitionFailed: 'Text recognition failed.',
        extractionOcrNoText: 'OCR found no text.',
        extractionOcrPageLimit: 'OCR page limit reached. Increase [ingest].ocr_max_pages and re-index to attempt this page.',
        documentOpenFailed: 'Could not open the current document: {detail}', folio: 'Folio', summoning: 'Summoning…',
        consulting: '— consulting the codex —', emptyFolio: '— (this folio is empty) —', folioResists: 'The folio resists.',
        noteLoadFailed: 'Could not retrieve note {id}: {detail}',
        sourceCurrent: 'Current source · the file still matches the indexed version.',
        sourceModified: 'Changed source · this is the indexed passage, but the file has changed. Reindex before treating it as current.',
        sourceMissing: 'Missing source · the file is no longer present; the indexed passage is preserved below.',
        sourceUnavailable: 'Unavailable source · Grimore cannot safely open the current file.',
        sourceUnverified: 'Unverified source · the index does not contain enough version information to compare the file.',
        sourceProvided: 'Provided source · this passage was attached explicitly for the question.',
        unreachable: 'unreachable', unknown: 'unknown', noStreamBody: 'The response stream was empty.',
    },
    es: {
        pageTitle: 'Grimore — Motor arcano de conocimiento', language: 'Idioma',
        setup: 'Configuración', setupEyebrow: 'Asistente inicial · solo lectura', closeSetupLabel: 'Cerrar configuración',
        setupTitle: 'Prepara tu biblioteca', closeSetup: 'Cerrar ✕',
        setupIntro: 'Valida la configuración activa, elige entre los modelos detectados y copia o aplica de forma segura los valores TOML resultantes.',
        setupInitial: 'Abre el asistente para validar esta instalación.',
        libraryFolder: 'Carpeta de la biblioteca',
        libraryFolderHelp: 'El servidor solo comprueba la carpeta actualmente activa. Edita este borrador, cópialo y reinicia Grimore.',
        chatModel: 'Modelo de conversación', embeddingModel: 'Modelo de embeddings', validation: 'Validación',
        scanEffects: 'Qué puede cambiar un escaneo', valuesFor: 'Valores para',
        configHelp: 'Sustituye las claves correspondientes en las secciones existentes; evita duplicar encabezados.',
        recheckConfig: 'Comprobar configuración', copyToml: 'Copiar valores TOML',
        activityEyebrow: 'Actividad de la biblioteca · solo lectura', indexingTitle: 'Registro de indexación',
        checking: 'Comprobando…', lookingForScan: 'Buscando un escaneo reciente…', noIndexingEvents: 'Aún no hay eventos de indexación.',
        indexingProgress: 'Progreso de indexación', indexingResults: 'Resultados de indexación',
        processed: 'Procesados', skipped: 'Omitidos', failed: 'Fallidos', previewed: 'Previsualizados', removed: 'Eliminados', chunks: 'Fragmentos',
        recentFiles: 'Archivos recientes', recovery: 'Recuperación',
        indexingNote: 'Las acciones protegidas requieren el token de API. La vista previa no modifica documentos; aplicar usa las protecciones normales del escaneo.',
        changeHistory: 'Cambios documentales aplicados',
        changeHistoryIntro: 'Las copias privadas previas permiten restaurar archivos solo mientras siga vigente la versión escrita por Grimore.',
        changeRetention: 'Se conservan los últimos 20 conjuntos de cambios',
        changeHistoryEmpty: 'No hay cambios documentales recuperables registrados.',
        changeFiles: '{count} archivo{suffix} modificado{suffix}', restoreFiles: 'Restaurar archivos',
        restoreConfirm: '¿Restaurar las copias anteriores a esta operación? Si hubo ediciones posteriores, se rechazará de forma segura.',
        restoringFiles: 'Restaurando las copias…', restoreComplete: 'Archivos restaurados. Ejecuta una vista previa antes de volver a aplicar la indexación.',
        restoreFailed: 'No se pudieron restaurar los archivos: {detail}',
        restoreConflict: 'La restauración se rechazó porque uno o más archivos cambiaron después.',
        previewChanges: 'Inspeccionar cambios propuestos', previewCreate: 'Crear', previewUpdate: 'Actualizar',
        previewTarget: 'Destino: {value}', beforeValue: 'Antes', afterValue: 'Después', emptyValue: '— vacío —',
        reviewEyebrow: 'Mantenimiento · solo candidatos', reviewInbox: 'Bandeja de revisión', show: 'Mostrar',
        reviewAll: 'Todos los candidatos', contradictions: 'Contradicciones', exactDuplicates: 'Duplicados exactos',
        nearDuplicates: 'Duplicados similares', reviewDue: 'Revisión pendiente', refreshInbox: 'Actualizar bandeja',
        reviewCandidateNotice: 'Son candidatos para revisión humana, no errores confirmados ni decisiones automáticas.',
        reviewCounts: 'Conteo de candidatos a revisión', reviewLoading: 'Cargando señales de mantenimiento…',
        reviewUnavailable: 'La bandeja de revisión no está disponible: {detail}', reviewEmpty: '— no hay candidatos en esta vista —',
        reviewShowing: 'Se muestran {count} candidato{suffix}.', candidate: 'Candidato', documents: 'Documentos',
        synthesisEyebrow: 'Borrador de conocimiento · sin escritura', topicSynthesis: 'Síntesis por tema',
        synthesisNeedsReview: 'Requiere revisión humana',
        synthesisIntro: 'Genera un borrador temporal a partir de pasajes indexados. No se guarda en la biblioteca.',
        synthesisKind: 'Agrupar por', synthesisTag: 'Etiqueta', synthesisCategory: 'Categoría',
        synthesisValue: 'Valor del tema', synthesisPlaceholder: 'p. ej., historia',
        synthesisPassagesPerNote: 'Pasajes por nota', generateSynthesis: 'Generar vista previa',
        synthesisIdle: 'Elige una etiqueta o categoría para inspeccionar un borrador de síntesis.',
        synthesisLoading: 'Construyendo un borrador revisable…',
        synthesisFailed: 'No se pudo generar la vista previa de síntesis: {detail}',
        synthesisNoOutput: 'No se produjo un borrador: {detail}', synthesisNoReason: 'sin resultado utilizable',
        synthesisReady: 'Borrador listo para revisar. Inspecciona su cobertura y los pasajes fuente antes de confiar en él.',
        synthesisPreviewOnly: 'Solo vista previa · no escrito', synthesisSources: 'Fuentes inspeccionables',
        synthesisNoSources: 'No hay pasajes fuente indexados disponibles para inspeccionar.',
        synthesisCoverageComplete: 'Completa para el alcance configurado',
        synthesisCoveragePartial: 'Cobertura incompleta', synthesisCoverageUnavailable: 'Cobertura no disponible',
        synthesisNotesMetric: '{included} de {eligible} notas aptas incluidas',
        synthesisPassagesMetric: '{selected} de {available} pasajes indexados seleccionados',
        synthesisContextMetric: '{used} de {limit} caracteres de contexto usados',
        synthesisSourceMetric: '{selected} de {available} pasajes seleccionados',
        synthesisSourceIncluded: 'Incluida en el borrador', synthesisSourceOmitted: 'Omitida por el límite de contexto',
        synthesisSourceUnavailable: 'Sin pasajes indexados',
        synthesisOpenSource: 'Inspeccionar documento fuente',
        synthesisLimitGenerated: 'Se excluyeron síntesis generadas anteriormente para evitar ciclos de retroalimentación.',
        synthesisLimitMissingPassages: 'Algunas notas coincidentes no tienen pasajes indexados y no pudieron usarse.',
        synthesisLimitContext: 'El límite de contexto omitió una o más notas que sí eran aptas.',
        synthesisLimitRepresentative: 'Solo se seleccionaron pasajes representativos; el texto indexado completo no se envió al modelo.',
        synthesisLimitUnknownCitations: 'El borrador contiene citas que no coinciden con una fuente incluida.',
        synthesisLimitUncitedSources: 'Una o más fuentes incluidas no aparecen citadas en el borrador.',
        synthesisLimitNoMatches: 'Ninguna nota indexada coincide con este tema.',
        exactDuplicateReason: '{count} documentos comparten el mismo contenido indexado.',
        nearDuplicateReason: 'Similitud semántica {score} (umbral {threshold}).',
        reviewDueReason: 'El plazo de revisión se superó por {days} día{suffix}.',
        contradictionFallback: 'Las afirmaciones indexadas podrían entrar en conflicto.',
        inspectCandidate: 'Inspeccionar evidencia y cobertura', reasonLabel: 'Por qué se señaló',
        analysisCoverage: 'Cobertura del análisis', involvedDocuments: 'Documentos implicados',
        candidatePassages: 'Pasajes para revisar', openDocument: 'Inspeccionar documento',
        noCandidatePassages: 'No hay un pasaje indexado disponible para este candidato.',
        coverageComplete: 'Completa para esta señal', coveragePartial: 'Parcial', coverageUnavailable: 'No disponible',
        coverageDocuments: '{analyzed} de {total} documentos analizados',
        coveragePassages: '{shown} de {available} pasajes de revisión mostrados',
        coverageChunks: '{count} fragmentos comparados',
        coverageClaimCap: 'La extracción analiza hasta {sections} partes de origen ({characters} caracteres en total) y devuelve hasta {claims} afirmaciones por documento.',
        coverageResultCap: 'Cada respuesta de la bandeja incluye como máximo {count} pares similares.',
        basisClaimPair: 'Un par de afirmaciones extraídas', basisContentHash: 'Huella del texto completo normalizado',
        basisMeanEmbeddings: 'Media de embeddings de fragmentos indexados', basisReviewWindow: 'Plazo configurado para la categoría',
        limitClaimPair: 'Solo se evaluó el par señalado, no todas las afirmaciones de ambos documentos.',
        limitClaimExtractionBounded: 'La extracción de afirmaciones está limitada intencionalmente en documentos largos.',
        limitModelCandidate: 'Un modelo propuso este conflicto; requiere confirmación humana.',
        limitRepresentativePassages: 'Los pasajes mostrados son representativos; la señal puede abarcar más texto indexado.',
        limitMetadataIgnored: 'Los metadatos del documento no participan en la comparación de contenido exacto.',
        limitSemanticNotIdentity: 'La cercanía semántica no demuestra duplicación ni equivalencia factual.',
        limitResultLimit: 'La bandeja muestra una cantidad acotada de los pares similares más fuertes.',
        limitAgeNotValidity: 'El tiempo transcurrido no demuestra que el contenido esté desactualizado.',
        extractedClaim: 'Afirmación extraída', indexedPassageLabel: 'Pasaje indexado',
        characterRange: 'Caracteres {start}–{end}',
        reviewActions: 'Decisión de revisión', confirmCandidate: 'Confirmar hallazgo', dismissCandidate: 'Descartar',
        postponeCandidate: 'Posponer', postponeFor: 'Posponer durante', days7: '7 días', days30: '30 días', days90: '90 días',
        reviewAuthHelp: 'Las decisiones requieren el token de operaciones protegidas de Configuración.',
        reviewDecisionSaving: 'Guardando la decisión de revisión…', reviewDecisionSaved: 'Decisión guardada en el historial.',
        reviewDecisionFailed: 'No se pudo guardar la decisión: {detail}',
        classificationTitle: 'Clasificación del documento', classificationCategory: 'Categoría',
        classificationTags: 'Etiquetas', classificationTagsHelp: 'Etiquetas separadas por comas',
        classificationManual: 'Corrección manual activa', classificationIndexed: 'Clasificación indexada actual',
        classificationSave: 'Guardar clasificación',
        classificationScope: 'Actualiza el índice y conserva esta elección en futuros escaneos; no edita ahora el documento fuente.',
        classificationSaving: 'Guardando categoría y etiquetas…',
        classificationSaved: 'Clasificación manual guardada; los futuros escaneos la conservarán.',
        classificationFailed: 'No se pudo guardar la clasificación: {detail}',
        documentDates: 'Fechas del documento', datePublished: 'Publicación', dateModified: 'Modificación',
        dateIndexed: 'Indexación', dateVerified: 'Última verificación', dateUnknown: 'Desconocida',
        dateSource: 'procedencia: {value}',
        dateSourceDocument: 'metadatos del documento ({field})', dateSourceFilesystem: 'fecha de modificación del archivo',
        dateSourceIndex: 'indexación de Grimore', dateSourceHuman: 'verificación humana',
        dateSourceLegacy: 'registro anterior (sin especificar)',
        decisionHistory: 'Historial de decisiones', historyEmpty: '— no hay decisiones de revisión registradas —',
        decisionConfirm: 'Confirmado', decisionDismiss: 'Descartado', decisionPostpone: 'Pospuesto',
        decisionRecordedAt: 'Registrado {value}', postponedUntil: 'hasta {value}',
        catalogue: 'Catálogo', catalogueHelp: 'Pulsa un encabezado para preguntar por su contenido.',
        conversationTitle: 'Memoria de conversación', newConversation: 'Nueva conversación',
        invocation: 'Consulta', questionPlaceholder: 'Formula una pregunta al códice…',
        searchScope: 'Alcance de búsqueda', category: 'Categoría', categoryHelp: 'Incluye las categorías descendientes.',
        tags: 'Etiquetas', tagsHelp: 'Separadas por comas; todas deben coincidir.', formats: 'Formatos',
        formatsHelp: 'Al elegir varios formatos puede coincidir cualquiera.', clearScope: 'Limpiar alcance',
        passages: 'pasajes', invoke: 'Consultar', toInvoke: 'para consultar', loadingFormats: 'Cargando formatos…', revelation: 'Respuesta',
        citedSources: 'Fuentes citadas', relatedPassages: 'Pasajes relacionados',
        marginalHelp: 'Las anotaciones marginales aparecen al realizar una consulta.',
        footerCreed: 'solo lectura · funciona sin conexión · modelos',
        allDocuments: 'Todos los documentos indexados', allCategories: 'Todas las categorías',
        noConversation: 'Sin turnos previos · la próxima pregunta inicia una conversación.',
        conversationHelp: 'Las tres respuestas más recientes darán contexto a las preguntas de seguimiento.',
        contextTurns: '{count} de {max} turnos de contexto conservados.', citedCount: '{count} fuente{suffix} citada{suffix}',
        scopeCategory: 'categoría: {value}', scopeTags: 'etiquetas: {value}', scopeFormats: 'formatos: {value}',
        scopeApplied: 'Este alcance se aplica a la respuesta y a los pasajes relacionados.',
        noPassagesYet: '— todavía no se han consultado pasajes —', noPassages: '— no se encontraron pasajes para esta pregunta —',
        noScopeMatches: '— ningún documento indexado coincide con el alcance —',
        requestWorking: 'Escribiendo…', requestFailed: 'La solicitud falló: {detail}',
        searchFailed: 'No se pudieron cargar los pasajes relacionados: {detail}', networkError: 'No se pudo contactar con el servidor.',
        noTokens: '— el códice no devolvió texto. ¿Está disponible el modelo? —',
        answerFailed: '— el códice guarda silencio. El servicio puede no estar disponible. —',
        inspectEvidence: 'Inspeccionar pasaje indexado y estado de la fuente',
        evidenceSupported: 'Evidencia disponible · revisa {count} pasaje{suffix} indexado{suffix}.',
        evidenceUnavailable: 'Se citaron fuentes, pero sus pasajes inspeccionables no están disponibles.',
        evidenceConflict: 'Evidencia contradictoria · compara los {count} pasajes antes de confiar en la respuesta.',
        evidenceUncertain: 'Respuesta incierta · revisa los pasajes y trata la conclusión con cautela.',
        evidenceInsufficient: 'Evidencia insuficiente · Grimore no recuperó respaldo suficiente para responder.',
        evidenceNoMatches: 'Evidencia insuficiente · ningún documento indexado coincide con el alcance.',
        evidenceError: 'Estado de evidencia no disponible · la solicitud no terminó correctamente.',
        evidencePdfPartial: 'Cobertura incompleta de PDF ({count}) · algunas páginas de los documentos citados no tenían texto extraíble al indexarse. La respuesta puede omitir información.',
        evidencePdfUnknown: 'Cobertura de PDF sin verificar ({count}) · no se registró el estado de extracción al indexar. Vuelve a escanear antes de confiar en afirmaciones sobre el documento completo.',
        setupDraft: 'Borrador modificado · copia estos valores y reinicia Grimore antes de comprobarlos.',
        currentSelection: 'selección actual', setupReady: 'Listo para un escaneo de prueba · no se ha realizado ningún cambio.',
        setupAttention: 'La configuración requiere atención · revisa las comprobaciones.', setupChecking: 'Comprobando carpeta y servicios de modelos…',
        setupFailed: 'Falló la validación: {detail}', copied: 'Copiado', copyUnavailable: 'No se pudo copiar',
        preview: 'Vista previa', apply: 'Aplicar', protected: 'Protegido',
        vaultCheckPass: 'La carpeta configurada es legible.', vaultCheckError: 'La carpeta configurada no existe, no es legible o no es un directorio.',
        vaultCheckFix: 'Actualiza [vault].path, reinicia Grimore y vuelve a validar.',
        chatCheckPass: 'El modelo de conversación “{model}” está disponible.', chatCheckError: 'El servicio no informó del modelo de conversación “{model}”.',
        chatCheckFix: 'Selecciona un modelo informado o instálalo en el servicio configurado.',
        embeddingCheckPass: 'El modelo de embeddings “{model}” está disponible.', embeddingCheckError: 'Ollama no informó del modelo de embeddings “{model}”.',
        embeddingCheckFix: 'Selecciona un modelo informado o descárgalo con Ollama.',
        configCheckWarning: 'No existe grimore.toml en el directorio de trabajo del servidor.', configCheckFix: 'Crea grimore.toml allí y reinicia Grimore.',
        scanPreviewRead: 'Lee los documentos modificados y envía el texto extraído al modelo de conversación.',
        scanPreviewShow: 'Muestra las etiquetas, el resumen y la categoría propuestos.', scanPreviewNoWrite: 'No reescribe documentos, archivos auxiliares ni notas indexadas.',
        scanApplyMarkdown: 'Escribe el frontmatter generado en los archivos Markdown modificados.', scanApplySidecars: 'Crea o actualiza archivos auxiliares administrados para formatos distintos de Markdown.',
        scanApplyIndex: 'Guarda metadatos, etiquetas, categorías, fragmentos y embeddings en SQLite.', scanApplyGit: 'Crea una instantánea de seguridad en Git antes de modificar documentos cuando está habilitado.',
        scanNeverOriginals: 'Nunca modifica el archivo PDF, EPUB, DOCX, ODT, RTF, HTML o TXT original.', scanNeverPrivate: 'Omite los documentos marcados con privacy: never_process.',
        writeToken: 'Token de operaciones protegidas', writeTokenHelp: 'Se conserva solo en la memoria de esta pestaña y se envía únicamente a operaciones de escritura protegidas.',
        saveConfig: 'Guardar configuración', writeDisabled: 'Las escrituras protegidas están desactivadas. Reinicia grimore serve con un token de API.',
        tokenRequired: 'Introduce el token de operaciones protegidas en Configuración.', configSaving: 'Guardando la configuración acotada…',
        applyConfigConfirm: '¿Guardar únicamente la ruta de la biblioteca y los modelos seleccionados en grimore.toml?',
        configSavedRestart: 'Configuración guardada. Reinicia Grimore para usar los valores nuevos.', configUnchanged: 'La configuración ya contiene estos valores.',
        previewIndexing: 'Iniciar vista previa segura', applyIndexing: 'Aplicar indexación', applyIndexingConfirm: '¿Aplicar la indexación ahora? Puede actualizar metadatos Markdown, archivos auxiliares, el índice e instantáneas Git.',
        indexingStarting: 'Iniciando la indexación…', indexingAccepted: 'Indexación aceptada; el progreso aparecerá aquí.',
        indexingFinished: 'El proceso de indexación terminó.', indexingExitFailed: 'El proceso de indexación terminó con código {code}. Revisa los detalles de recuperación.',
        restartBeforeIndexing: 'Reinicia Grimore antes de indexar con la configuración guardada.',
        operationFailed: 'Falló la operación protegida: {detail}', invalidToken: 'El token de operaciones protegidas no es válido.', indexingActive: 'Ya hay una operación de indexación en curso.',
        indexingRunning: 'Indexando', indexingComplete: 'Completado', indexingErrors: 'Completado con errores',
        indexingInterrupted: 'Interrumpido', indexingStopped: 'Vigilante detenido', noRun: 'Sin ejecuciones registradas',
        previewFirst: 'Ejecuta grimore scan --dry-run para probar la primera indexación.',
        nowReading: 'Leyendo: {path}', filesExamined: '{completed} de {total} archivos examinados.',
        watcherReady: 'El vigilante está listo para detectar cambios.', statusUnavailable: 'Estado no disponible',
        indexingUnavailable: 'No se pudo leer la actividad de indexación: {detail}', noFileResults: 'Todavía no hay resultados de archivos para esta biblioteca.',
        unknownFile: 'Archivo desconocido', noRecovery: 'No se necesita ninguna acción de recuperación.',
        noFormats: 'No hay formatos configurados.', scopeUnavailable: 'Las opciones de alcance no están disponibles: {detail}', formatsUnavailable: 'Formatos no disponibles.',
        catalogueUnavailable: '— catálogo no disponible —', noCategories: '— no hay categorías indexadas —', uncategorised: '— sin categoría —',
        categoryQuestion: '¿Qué contiene el códice sobre {category}?', marginalGloss: 'Glosa marginal', note: 'nota', score: 'puntuación',
        enterQuestion: 'Escribe una pregunta antes de consultar el códice.', noMatchesHelp: 'Ningún documento indexado coincide con este alcance. Limpia o ajusta los filtros.',
        citationsRemoved: 'Se eliminaron {count} cita{suffix} del modelo porque no pertenecían al contexto recuperado.',
        close: 'Cerrar', sourceEvidence: 'Evidencia de la fuente', retrievedPassages: 'Pasajes recuperados', previous: 'Anterior', next: 'Siguiente',
        chooseCitationPassage: 'Elegir un pasaje',
        ambiguousCitation: 'Esta referencia coincide con varios pasajes recuperados. Elige uno para inspeccionarlo; la cita no identifica un pasaje único.',
        unverifiableSelection: 'No se puede verificar el pasaje elegido a partir de esta respuesta. Estos son pasajes recuperados coincidentes, no una selección confirmada del modelo.',
        selectedCitation: 'Inspeccionar el pasaje elegido por esta cita',
        chooseCitation: 'Elegir entre los pasajes coincidentes',
        indexedPassage: 'Pasaje indexado', currentDocument: 'Documento actual · pasaje resaltado', goTo: 'Ir a página o sección',
        locatingPassage: '— localizando el pasaje —', source: 'Fuente', noDocumentPath: 'Sin ruta de documento actual',
        noIndexedPassage: 'No hay un pasaje indexado disponible.', page: 'Página {value}', section: 'Sección: {value}', passage: 'Pasaje {value}',
        excerptOnly: 'Solo extracto', noLocator: 'No hay localizador de página o sección', position: '{current} de {total}',
        passageNotLocated: 'No se pudo localizar el pasaje exacto en la representación actual; usa el pasaje indexado superior.',
        extractionPartial: 'La extracción está incompleta. No se recuperó texto de la(s) página(s) {pages}; las respuestas pueden omitir información allí.',
        extractionRecovered: 'OCR o un método alternativo recuperó texto de la(s) página(s) {pages} del PDF actual. Comprueba esos pasajes en el original; las respuestas indexadas pueden necesitar una actualización.',
        extractionFailed: 'No se pudo leer el documento actual. El pasaje indexado superior puede estar desactualizado o incompleto.',
        extractionOcrUnavailable: 'Las herramientas de OCR no están disponibles.',
        extractionOcrRenderFailed: 'No se pudo renderizar la página para OCR.',
        extractionOcrRecognitionFailed: 'Falló el reconocimiento de texto.',
        extractionOcrNoText: 'OCR no encontró texto.',
        extractionOcrPageLimit: 'Se alcanzó el límite de páginas OCR. Aumenta [ingest].ocr_max_pages y vuelve a indexar para intentar esta página.',
        documentOpenFailed: 'No se pudo abrir el documento actual: {detail}', folio: 'Folio', summoning: 'Invocando…',
        consulting: '— consultando el códice —', emptyFolio: '— (este folio está vacío) —', folioResists: 'El folio se resiste.',
        noteLoadFailed: 'No se pudo recuperar la nota {id}: {detail}',
        sourceCurrent: 'Fuente vigente · el archivo aún coincide con la versión indexada.',
        sourceModified: 'Fuente modificada · este es el pasaje indexado, pero el archivo cambió. Reindexa antes de tratarlo como vigente.',
        sourceMissing: 'Fuente ausente · el archivo ya no está presente; se conserva el pasaje indexado.',
        sourceUnavailable: 'Fuente no disponible · Grimore no puede abrir el archivo actual de forma segura.',
        sourceUnverified: 'Fuente no verificada · el índice no tiene información suficiente para comparar el archivo.',
        sourceProvided: 'Fuente aportada · este pasaje se adjuntó explícitamente a la pregunta.',
        unreachable: 'no disponible', unknown: 'desconocido', noStreamBody: 'El flujo de respuesta llegó vacío.',
    },
};

function t(key, values = {}) {
    const table = I18N[state.language] || I18N.en;
    let value = table[key] ?? I18N.en[key] ?? key;
    for (const [name, replacement] of Object.entries(values)) {
        value = value.replaceAll(`{${name}}`, String(replacement));
    }
    return value;
}

function renderExtractionWarning(extraction) {
    if (!extraction || !['partial', 'recovered', 'failed'].includes(extraction.status)) return null;
    const issues = Array.isArray(extraction.issues) ? extraction.issues : [];
    const relevant = extraction.status === 'partial'
        ? issues.filter(issue => !issue.recovered)
        : issues.filter(issue => issue.recovered);
    const pages = [...new Set(relevant.map(issue => issue.page).filter(Number.isInteger))];
    const warning = document.createElement('p');
    warning.className = 'drawer__extraction-warning';
    warning.setAttribute('role', 'status');
    const key = extraction.status === 'failed' ? 'extractionFailed'
        : extraction.status === 'partial' ? 'extractionPartial' : 'extractionRecovered';
    warning.textContent = t(key, { pages: pages.join(', ') || '—' });
    if (extraction.status === 'partial') {
        const reasonKeys = {
            ocr_unavailable: 'extractionOcrUnavailable',
            ocr_render_failed: 'extractionOcrRenderFailed',
            ocr_recognition_failed: 'extractionOcrRecognitionFailed',
            ocr_no_text: 'extractionOcrNoText',
            ocr_page_limit: 'extractionOcrPageLimit',
        };
        const reasons = [...new Set(relevant.map(issue => reasonKeys[issue.reason]).filter(Boolean))];
        if (reasons.length) warning.textContent += ` ${reasons.map(reason => t(reason)).join(' ')}`;
    }
    return warning;
}

function apiFetch(resource, options = {}) {
    const headers = new Headers(options.headers || {});
    if (state.writeToken) headers.set('Authorization', `Bearer ${state.writeToken}`);
    return fetch(resource, { ...options, headers });
}

function preferredLanguage() {
    try {
        const saved = window.localStorage.getItem('grimore-language');
        if (saved === 'en' || saved === 'es') return saved;
    } catch { /* preference storage is optional */ }
    return String(navigator.language || 'en').toLowerCase().startsWith('es') ? 'es' : 'en';
}

function applyLanguage(language, { persist = true } = {}) {
    state.language = language === 'es' ? 'es' : 'en';
    document.documentElement.lang = state.language;
    document.title = t('pageTitle');
    $('#language-select').value = state.language;
    $$('[data-i18n]').forEach(element => {
        element.textContent = t(element.dataset.i18n);
    });
    $$('[data-i18n-placeholder]').forEach(element => {
        element.placeholder = t(element.dataset.i18nPlaceholder);
    });
    $$('[data-i18n-aria-label]').forEach(element => {
        element.setAttribute('aria-label', t(element.dataset.i18nAriaLabel));
    });
    const allCategories = $('#filter-category option[value=""]');
    if (allCategories) allCategories.textContent = t('allCategories');
    renderConversation();
    updateFilterSummary();
    renderCategories();
    renderHits(state.lastSearchHits, state.retrievalScope);
    renderMarginalia(state.lastSearchHits);
    if (state.setup) renderSetup(state.setup, { preserveDraft: true });
    if (state.sourceLabels.length) renderSources(state.sourceLabels, state.sourceDetails);
    wireCitations();
    if (state.evidenceStatus) {
        renderEvidenceStatus(state.evidenceStatus, state.sourceDetails, state.retrievalScope);
    }
    if (state.indexing) renderIndexing(state.indexing);
    if (state.reviewInbox) renderReviewInbox();
    if (state.synthesisPreview) renderSynthesisPreview();
    if (persist) {
        try { window.localStorage.setItem('grimore-language', state.language); } catch { /* optional */ }
    }
}

// ── Sanitisation helpers ─────────────────────────────────────────

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));
}

// Convert raw streamed text into HTML with [[citations]] turned into
// clickable spans. Always called on full collected text so the parser
// sees complete citation tokens.
const CITE_RE = /\[\[([^\[\]]+)\]\]/g;

function renderCitedHtml(text, counter = { index: 0 }) {
    let offset = 0;
    let html = '';
    for (const match of text.matchAll(CITE_RE)) {
        html += escapeHtml(text.slice(offset, match.index));
        const label = match[1];
        const safeLabel = escapeHtml(label).replace(/\r/g, '&#13;').replace(/\n/g, '&#10;');
        const safeAttribute = escapeHtml(label.trim()).replace(/\r/g, '&#13;').replace(/\n/g, '&#10;');
        html += `<span class="cite" data-label="${safeAttribute}" data-citation-index="${counter.index++}">${safeLabel}</span>`;
        offset = match.index + match[0].length;
    }
    return html + escapeHtml(text.slice(offset));
}

function renderProse(raw) {
    const paragraphs = renderCitedHtml(raw).split(/\n{2,}/).filter(p => p.trim());
    if (paragraphs.length === 0) return '';
    return paragraphs.map(p =>
        `<p>${p.replace(/\n/g, '<br>')}</p>`
    ).join('');
}

// Lower-roman for marginalia indices: i, ii, iii…
function romanize(n) {
    const map = ['', 'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'];
    return map[n] || String(n);
}

function makeKeyboardActivatable(element, handler, role = 'button') {
    element.tabIndex = 0;
    element.setAttribute('role', role);
    element.addEventListener('click', handler);
    element.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            handler();
        }
    });
}

// ── Health (the candle) ─────────────────────────────────────────

async function refreshHealth() {
    const el = $('#health');
    try {
        const r = await apiFetch('/api/health', { cache: 'no-store' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        state.health = data;
        const mode = data.cognition?.mode || 'unknown';
        const modeEl = $('#model-mode');
        if (modeEl) modeEl.textContent = mode;
        if (data.ok) {
            el.classList.add('is-ok');
            el.classList.remove('is-err');
            $('.health__label', el).textContent = `v${data.version} · ${mode}`;
        } else {
            throw new Error('unhealthy');
        }
    } catch (e) {
        el.classList.add('is-err');
        el.classList.remove('is-ok');
        $('.health__label', el).textContent = t('unreachable');
        const modeEl = $('#model-mode');
        if (modeEl) modeEl.textContent = t('unknown');
    }
}

// ── Read-only setup assistant ──────────────────────────────────

function tomlString(value) {
    return JSON.stringify(String(value || ''));
}

function setupSnippet() {
    return [
        '[vault]',
        `path = ${tomlString($('#setup-vault').value.trim())}`,
        '',
        '[cognition]',
        `model_llm_local = ${tomlString($('#setup-chat').value)}`,
        `model_embeddings_local = ${tomlString($('#setup-embedding').value)}`,
    ].join('\n');
}

function updateSetupSnippet() {
    $('#setup-snippet').textContent = setupSnippet();
    if (!state.setup) return;
    const changed = (
        $('#setup-vault').value.trim() !== state.setup.vault.path ||
        $('#setup-chat').value !== state.setup.models.chat.selected ||
        $('#setup-embedding').value !== state.setup.models.embeddings.selected
    );
    if (changed) {
        const status = $('#setup-status');
        status.className = 'setup-status is-warning';
        status.textContent = t('setupDraft');
    }
}

function fillModelSelect(select, selected, available) {
    select.innerHTML = '';
    const availableNames = available || [];
    const names = [...new Set([selected, ...availableNames].filter(Boolean))];
    for (const name of names) {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name === selected && !availableNames.includes(name)
            ? `${name} · ${t('currentSelection')}`
            : name;
        select.appendChild(option);
    }
    select.value = selected;
}

function localisedSetupCheck(check, data) {
    const selected = check.id === 'chat_model'
        ? data.models.chat.selected
        : data.models.embeddings.selected;
    const keys = {
        vault: [check.status === 'pass' ? 'vaultCheckPass' : 'vaultCheckError', 'vaultCheckFix'],
        chat_model: [check.status === 'pass' ? 'chatCheckPass' : 'chatCheckError', 'chatCheckFix'],
        embedding_model: [check.status === 'pass' ? 'embeddingCheckPass' : 'embeddingCheckError', 'embeddingCheckFix'],
        config_file: ['configCheckWarning', 'configCheckFix'],
    };
    const [messageKey, fixKey] = keys[check.id] || [];
    return messageKey
        ? { message: t(messageKey, { model: selected }), fix: t(fixKey) }
        : { message: check.message, fix: check.fix };
}

const SETUP_EFFECT_KEYS = {
    preview: ['scanPreviewRead', 'scanPreviewShow', 'scanPreviewNoWrite'],
    apply: ['scanApplyMarkdown', 'scanApplySidecars', 'scanApplyIndex', 'scanApplyGit'],
    never: ['scanNeverOriginals', 'scanNeverPrivate'],
};

function renderSetup(data, { preserveDraft = false } = {}) {
    const draft = preserveDraft ? {
        vault: $('#setup-vault').value,
        chat: $('#setup-chat').value,
        embedding: $('#setup-embedding').value,
    } : null;
    state.setup = data;
    state.writeEnabled = Boolean(data.operations?.write_enabled);
    state.restartRequired = Boolean(data.operations?.restart_required);
    $('#setup-vault').value = draft?.vault ?? data.vault.path ?? '';
    fillModelSelect($('#setup-chat'), draft?.chat || data.models.chat.selected, data.models.chat.available);
    fillModelSelect(
        $('#setup-embedding'),
        draft?.embedding || data.models.embeddings.selected,
        data.models.embeddings.available,
    );

    const status = $('#setup-status');
    status.className = `setup-status ${data.ready ? 'is-ready' : 'is-error'}`;
    status.textContent = data.ready
        ? t('setupReady')
        : t('setupAttention');

    const checks = $('#setup-checks');
    checks.innerHTML = '';
    for (const check of data.checks || []) {
        const item = document.createElement('li');
        item.className = `setup-check setup-check--${check.status}`;
        const copy = localisedSetupCheck(check, data);
        const message = document.createElement('strong');
        message.textContent = copy.message;
        const fix = document.createElement('span');
        fix.textContent = check.status === 'pass' ? '' : copy.fix;
        item.append(message, fix);
        checks.appendChild(item);
    }

    const effects = $('#setup-effects');
    effects.innerHTML = '';
    const groups = [
        [t('preview'), 'preview'],
        [t('apply'), 'apply'],
        [t('protected'), 'never'],
    ];
    for (const [label, group] of groups) {
        const details = document.createElement('details');
        details.open = label === t('apply');
        const summary = document.createElement('summary');
        summary.textContent = label;
        const list = document.createElement('ul');
        const lines = SETUP_EFFECT_KEYS[group]?.map(key => t(key))
            || data.scan_effects[group]
            || [];
        for (const line of lines) {
            const item = document.createElement('li');
            item.textContent = line;
            list.appendChild(item);
        }
        details.append(summary, list);
        effects.appendChild(details);
    }
    updateSetupSnippet();
    updateProtectedControls();
}

async function loadSetup() {
    const status = $('#setup-status');
    status.className = 'setup-status is-loading';
    status.textContent = t('setupChecking');
    try {
        const response = await apiFetch('/api/setup', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderSetup(await response.json());
        if (state.writeToken) {
            refreshHealth();
            loadCategories();
            loadFilterOptions();
            refreshIndexingOnce();
            loadReviewInbox();
        }
    } catch (error) {
        status.className = 'setup-status is-error';
        status.textContent = t('setupFailed', { detail: error.message });
    }
}

function openSetup() {
    const panel = $('#setup-panel');
    panel.hidden = false;
    $('#setup-toggle').setAttribute('aria-expanded', 'true');
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    loadSetup();
}

function closeSetup() {
    $('#setup-panel').hidden = true;
    $('#setup-toggle').setAttribute('aria-expanded', 'false');
    $('#setup-toggle').focus();
}

async function copySetupSnippet() {
    const button = $('#setup-copy');
    try {
        await navigator.clipboard.writeText(setupSnippet());
        button.textContent = t('copied');
    } catch {
        button.textContent = t('copyUnavailable');
    }
    setTimeout(() => { button.textContent = t('copyToml'); }, 1800);
}

function protectedHeaders() {
    return {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${state.writeToken}`,
    };
}

function protectedError(error, status) {
    if (status === 401) return t('invalidToken');
    return t('operationFailed', { detail: friendlyError(error) });
}

function updateProtectedControls() {
    const authorised = state.writeEnabled && Boolean(state.writeToken);
    const indexingActive = Boolean(
        state.indexing?.active || state.indexing?.operation?.active,
    );
    $('#setup-apply').disabled = !authorised || state.operationBusy;
    $('#indexing-preview').disabled = !authorised || state.operationBusy || indexingActive || state.restartRequired;
    $('#indexing-apply').disabled = !authorised || state.operationBusy || indexingActive || state.restartRequired;
    $$('.review-action').forEach(control => {
        control.disabled = !authorised || state.reviewDecisionBusy || state.reviewClassificationBusy;
    });
    $$('.classification-action').forEach(control => {
        control.disabled = !authorised || state.reviewDecisionBusy || state.reviewClassificationBusy;
    });
    $$('.restore-action').forEach(control => {
        control.disabled = !authorised || state.operationBusy || indexingActive;
    });
    const status = $('#indexing-operation-status');
    if (!state.writeEnabled) {
        status.textContent = t('writeDisabled');
        status.dataset.guidance = 'true';
    } else if (!state.writeToken) {
        status.textContent = t('tokenRequired');
        status.dataset.guidance = 'true';
    } else if (state.restartRequired) {
        status.textContent = t('restartBeforeIndexing');
        status.dataset.guidance = 'true';
    } else if (status.dataset.guidance === 'true') {
        status.textContent = indexingActive ? t('indexingActive') : '';
        status.dataset.guidance = 'false';
    }
    const reviewStatus = $('#review-operation-status');
    if (!state.writeEnabled) {
        reviewStatus.textContent = t('writeDisabled');
        reviewStatus.dataset.guidance = 'true';
    } else if (!state.writeToken) {
        reviewStatus.textContent = t('reviewAuthHelp');
        reviewStatus.dataset.guidance = 'true';
    } else if (reviewStatus.dataset.guidance === 'true') {
        reviewStatus.textContent = '';
        reviewStatus.dataset.guidance = 'false';
    }
}

async function applySetupConfig() {
    if (!state.writeEnabled || !state.writeToken || state.operationBusy) {
        updateProtectedControls();
        return;
    }
    if (!window.confirm(t('applyConfigConfirm'))) return;
    state.operationBusy = true;
    updateProtectedControls();
    const status = $('#setup-status');
    status.className = 'setup-status is-loading';
    status.textContent = t('configSaving');
    try {
        const response = await apiFetch('/api/setup/apply', {
            method: 'POST',
            headers: protectedHeaders(),
            body: JSON.stringify({
                vault_path: $('#setup-vault').value.trim(),
                chat_model: $('#setup-chat').value,
                embedding_model: $('#setup-embedding').value,
                confirm: 'apply-config',
            }),
        });
        if (!response.ok) throw Object.assign(
            new Error(await responseError(response)), { status: response.status },
        );
        const result = await response.json();
        state.restartRequired = Boolean(result.restart_required);
        status.className = `setup-status ${result.changed ? 'is-warning' : 'is-ready'}`;
        status.textContent = result.changed ? t('configSavedRestart') : t('configUnchanged');
    } catch (error) {
        status.className = 'setup-status is-error';
        status.textContent = protectedError(error, error.status);
    } finally {
        state.operationBusy = false;
        updateProtectedControls();
    }
}

async function startIndexing(mode) {
    if (!state.writeEnabled || !state.writeToken || state.operationBusy) {
        updateProtectedControls();
        return;
    }
    if (mode === 'apply' && !window.confirm(t('applyIndexingConfirm'))) return;
    state.operationBusy = true;
    updateProtectedControls();
    const status = $('#indexing-operation-status');
    status.dataset.guidance = 'false';
    status.textContent = t('indexingStarting');
    try {
        const response = await apiFetch('/api/indexing/start', {
            method: 'POST',
            headers: protectedHeaders(),
            body: JSON.stringify({
                mode,
                ...(mode === 'apply' ? { confirm: 'apply-indexing' } : {}),
            }),
        });
        if (!response.ok) throw Object.assign(
            new Error(await responseError(response)), { status: response.status },
        );
        const result = await response.json();
        state.indexing = {
            ...(state.indexing || {}), active: true, operation: result.operation,
        };
        status.textContent = t('indexingAccepted');
        window.setTimeout(refreshIndexingOnce, 350);
    } catch (error) {
        status.textContent = error.status === 409
            ? t('indexingActive')
            : protectedError(error, error.status);
    } finally {
        state.operationBusy = false;
        updateProtectedControls();
    }
}

// ── Indexing activity ─────────────────────────────────────────

const INDEXING_STATUS = {
    running: 'indexingRunning',
    completed: 'indexingComplete',
    completed_with_errors: 'indexingErrors',
    interrupted: 'indexingInterrupted',
    stopped: 'indexingStopped',
};

const RECOVERY_ES = {
    'start-preview': ['Sin ejecuciones de indexación', 'Ejecuta primero una vista previa segura para validar el proceso.'],
    'resume-interrupted': ['Una indexación se detuvo inesperadamente', 'Corrige la causa indicada y repite el escaneo; los archivos completados se omitirán rápidamente.'],
    'retry-failed': ['Algunos archivos fallaron', 'Revisa los archivos fallidos. Un nuevo escaneo reintenta los incompletos.'],
    'review-privacy': ['Hay archivos excluidos por privacidad', 'Mantén la exclusión o elimina privacy: never_process de forma intencional.'],
};

function renderIndexingItems(runs) {
    const list = $('#indexing-items');
    list.innerHTML = '';
    const items = (runs || [])
        .flatMap(run => (run.items || []).map(item => ({ ...item, source: run.source })))
        .sort((left, right) => String(right.ts || '').localeCompare(String(left.ts || '')))
        .slice(0, 6);

    if (items.length === 0) {
        const empty = document.createElement('li');
        empty.className = 'indexing-empty';
        empty.textContent = t('noFileResults');
        list.appendChild(empty);
        return;
    }

    for (const item of items) {
        const row = document.createElement('li');
        row.className = `indexing-item indexing-item--${item.outcome}`;
        const badge = document.createElement('span');
        badge.className = 'indexing-item__outcome';
        badge.textContent = I18N[state.language]?.[item.outcome] || item.outcome;
        const content = document.createElement('div');
        const path = document.createElement('strong');
        path.textContent = item.path || t('unknownFile');
        content.appendChild(path);
        const detailText = item.error || item.reason;
        if (detailText) {
            const detail = document.createElement('small');
            detail.textContent = detailText;
            content.appendChild(detail);
        }
        if (item.changes) {
            const details = document.createElement('details');
            details.className = 'indexing-change-preview';
            const summary = document.createElement('summary');
            summary.textContent = t('previewChanges');
            details.appendChild(summary);
            const target = document.createElement('p');
            const actionKey = item.changes.action === 'create'
                ? 'previewCreate' : 'previewUpdate';
            target.textContent = `${t(actionKey)} · ${t('previewTarget', {
                value: item.changes.target || item.path,
            })}`;
            details.appendChild(target);
            const fields = document.createElement('dl');
            for (const field of item.changes.fields || []) {
                const row = document.createElement('div');
                const name = document.createElement('dt');
                name.textContent = field.name;
                const values = document.createElement('dd');
                const displayValue = value => {
                    if (value == null || value === '' || (Array.isArray(value) && !value.length)) {
                        return t('emptyValue');
                    }
                    return Array.isArray(value) ? value.join(', ') : String(value);
                };
                values.textContent = `${t('beforeValue')}: ${displayValue(field.before)} · ${t('afterValue')}: ${displayValue(field.after)}`;
                row.append(name, values);
                fields.appendChild(row);
            }
            details.appendChild(fields);
            content.appendChild(details);
        }
        row.append(badge, content);
        list.appendChild(row);
    }
}

function renderChangeHistory(changes) {
    const list = $('#change-history-items');
    list.innerHTML = '';
    const visible = (changes || []).filter(change => change.file_count > 0).slice(0, 8);
    if (!visible.length) {
        const empty = document.createElement('li');
        empty.className = 'indexing-empty';
        empty.textContent = t('changeHistoryEmpty');
        list.appendChild(empty);
        return;
    }
    for (const change of visible) {
        const item = document.createElement('li');
        item.className = `change-history__item change-history__item--${change.status}`;
        const head = document.createElement('div');
        const title = document.createElement('strong');
        title.textContent = change.operation || 'Grimore';
        const when = document.createElement('time');
        when.textContent = reviewDate(change.finished_at || change.started_at);
        head.append(title, when);
        const count = document.createElement('span');
        count.textContent = t('changeFiles', {
            count: change.file_count,
            suffix: change.file_count === 1 ? '' : 's',
        });
        const files = document.createElement('ul');
        for (const file of change.files || []) {
            const fileItem = document.createElement('li');
            fileItem.textContent = `${file.action === 'created' ? t('previewCreate') : t('previewUpdate')} · ${file.path}`;
            files.appendChild(fileItem);
        }
        item.append(head, count, files);
        if (change.restorable) {
            const restore = document.createElement('button');
            restore.type = 'button';
            restore.className = 'restore-action';
            restore.textContent = t('restoreFiles');
            restore.addEventListener('click', () => restoreChangeSet(change.id));
            item.appendChild(restore);
        }
        list.appendChild(item);
    }
    updateProtectedControls();
}

async function restoreChangeSet(changeId) {
    if (!state.writeEnabled || !state.writeToken || state.operationBusy) {
        updateProtectedControls();
        return;
    }
    if (!window.confirm(t('restoreConfirm'))) return;
    state.operationBusy = true;
    updateProtectedControls();
    const status = $('#change-operation-status');
    status.textContent = t('restoringFiles');
    try {
        const response = await apiFetch(
            `/api/changes/${encodeURIComponent(changeId)}/restore`,
            {
                method: 'POST',
                headers: protectedHeaders(),
                body: JSON.stringify({ confirm: 'restore-change-set' }),
            },
        );
        if (!response.ok) {
            const detail = await responseError(response);
            throw Object.assign(new Error(detail), { status: response.status });
        }
        status.textContent = t('restoreComplete');
        await refreshIndexingOnce();
    } catch (error) {
        status.textContent = error.status === 409
            && String(error.message || '').includes('changed afterwards')
            ? t('restoreConflict')
            : t('restoreFailed', { detail: protectedError(error, error.status) });
    } finally {
        state.operationBusy = false;
        updateProtectedControls();
    }
}

function renderIndexingActions(actions) {
    const list = $('#indexing-actions');
    list.innerHTML = '';
    if (!actions || actions.length === 0) {
        const item = document.createElement('li');
        item.className = 'indexing-empty';
        item.textContent = t('noRecovery');
        list.appendChild(item);
        return;
    }
    for (const action of actions) {
        const item = document.createElement('li');
        const title = document.createElement('strong');
        const localized = state.language === 'es' ? RECOVERY_ES[action.id] : null;
        title.textContent = localized?.[0] || action.title;
        const detail = document.createElement('span');
        detail.textContent = localized?.[1] || action.detail;
        item.append(title, detail);
        if (action.command) {
            const command = document.createElement('code');
            command.textContent = action.command;
            item.appendChild(command);
        }
        list.appendChild(item);
    }
}

function renderIndexing(data) {
    state.indexing = data;
    if (typeof data.operation?.write_enabled === 'boolean') {
        state.writeEnabled = data.operation.write_enabled;
    }
    const operationStatus = $('#indexing-operation-status');
    if (data.operation?.active && operationStatus.dataset.guidance !== 'true') {
        operationStatus.textContent = t('indexingAccepted');
    } else if (data.operation?.exit_code != null) {
        operationStatus.dataset.guidance = 'false';
        operationStatus.textContent = data.operation.exit_code === 0
            ? t('indexingFinished')
            : t('indexingExitFailed', { code: data.operation.exit_code });
    }
    updateProtectedControls();
    const run = data.latest;
    const status = $('#indexing-state');
    const progress = $('#indexing-progress');
    const bar = $('#indexing-progress-bar');
    progress.classList.remove('is-indeterminate');

    if (!run) {
        status.className = 'indexing-state';
        status.textContent = t('noRun');
        progress.removeAttribute('aria-valuenow');
        bar.style.width = '0%';
        $('#indexing-current').textContent = t('previewFirst');
        for (const id of ['processed', 'skipped', 'failed', 'chunks']) {
            $(`#indexing-${id}`).textContent = '0';
        }
        renderIndexingItems([]);
        renderIndexingActions(data.recovery_actions);
        renderChangeHistory(data.changes);
        return;
    }

    const label = INDEXING_STATUS[run.status]
        ? t(INDEXING_STATUS[run.status])
        : (run.status || 'Unknown');
    status.className = `indexing-state indexing-state--${run.status || 'unknown'}`;
    status.textContent = `${label} · ${run.source || 'indexer'}${run.dry_run ? ' preview' : ''}`;

    if (typeof run.progress === 'number') {
        const percent = Math.round(run.progress * 100);
        progress.setAttribute('aria-valuenow', String(percent));
        bar.style.width = `${percent}%`;
    } else {
        progress.removeAttribute('aria-valuenow');
        bar.style.width = run.status === 'running' ? '35%' : '100%';
        if (run.status === 'running') progress.classList.add('is-indeterminate');
    }

    const counts = run.counts || {};
    for (const id of ['processed', 'skipped', 'failed', 'chunks']) {
        $(`#indexing-${id}`).textContent = String(counts[id] || 0);
    }
    if (run.current_path) {
        $('#indexing-current').textContent = t('nowReading', { path: run.current_path });
    } else if (typeof run.total === 'number') {
        $('#indexing-current').textContent = t('filesExamined', {
            completed: counts.completed || 0, total: run.total,
        });
    } else {
        $('#indexing-current').textContent = t('watcherReady');
    }
    renderIndexingItems(data.runs);
    renderIndexingActions(data.recovery_actions);
    renderChangeHistory(data.changes);
}

async function refreshIndexingOnce() {
    try {
        const response = await apiFetch('/api/indexing', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderIndexing(await response.json());
    } catch (error) {
        $('#indexing-operation-status').textContent = t('indexingUnavailable', {
            detail: friendlyError(error),
        });
    }
}

async function loadIndexing() {
    let delay = 10_000;
    try {
        const response = await apiFetch('/api/indexing', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        renderIndexing(data);
        delay = data.active ? 1_500 : 10_000;
    } catch (error) {
        const status = $('#indexing-state');
        status.className = 'indexing-state indexing-state--interrupted';
        status.textContent = t('statusUnavailable');
        $('#indexing-current').textContent = t('indexingUnavailable', { detail: error.message });
    }
    window.setTimeout(loadIndexing, delay);
}

// ── Review inbox ───────────────────────────────────────────────

const REVIEW_KIND_LABELS = {
    contradiction: 'contradictions',
    duplicate_exact: 'exactDuplicates',
    duplicate_near: 'nearDuplicates',
    review_due: 'reviewDue',
};

const REVIEW_COVERAGE_BASIS = {
    extracted_claim_pair: 'basisClaimPair',
    normalized_content_hash: 'basisContentHash',
    mean_chunk_embeddings: 'basisMeanEmbeddings',
    category_review_window: 'basisReviewWindow',
};

const REVIEW_LIMITATIONS = {
    claim_pair_only: 'limitClaimPair',
    claim_extraction_bounded: 'limitClaimExtractionBounded',
    model_candidate: 'limitModelCandidate',
    representative_passages_only: 'limitRepresentativePassages',
    metadata_ignored: 'limitMetadataIgnored',
    semantic_similarity_not_identity: 'limitSemanticNotIdentity',
    result_limit_applied: 'limitResultLimit',
    age_not_content_validity: 'limitAgeNotValidity',
};

function reviewReason(item) {
    const metadata = item.metadata || {};
    if (item.kind === 'duplicate_exact') {
        const count = metadata.copies || (item.documents || []).length;
        return t('exactDuplicateReason', { count });
    }
    if (item.kind === 'duplicate_near') {
        return t('nearDuplicateReason', {
            score: Number(metadata.score || 0).toLocaleString(state.language, {
                maximumFractionDigits: 3,
            }),
            threshold: Number(metadata.threshold || 0).toLocaleString(state.language, {
                maximumFractionDigits: 3,
            }),
        });
    }
    if (item.kind === 'review_due') {
        const days = Number(metadata.days_overdue || 0);
        return t('reviewDueReason', {
            days,
            suffix: days === 1 ? '' : 's',
        });
    }
    return item.reason || t('contradictionFallback');
}

function reviewCoverageStatus(status) {
    return t({
        complete: 'coverageComplete',
        partial: 'coveragePartial',
        unavailable: 'coverageUnavailable',
    }[status] || 'coverageUnavailable');
}

function reviewPassageLocator(passage) {
    const parts = [];
    if (passage.source === 'extracted_claim') parts.push(t('extractedClaim'));
    else parts.push(t('indexedPassageLabel'));
    if (passage.page != null) parts.push(t('page', { value: passage.page }));
    if (passage.heading) parts.push(t('section', { value: passage.heading }));
    if (Number.isInteger(passage.char_start) && Number.isInteger(passage.char_end)) {
        parts.push(t('characterRange', { start: passage.char_start, end: passage.char_end }));
    }
    if (passage.complete === false) parts.push(t('excerptOnly'));
    return parts.join(' · ');
}

function reviewSectionTitle(key) {
    const title = document.createElement('h4');
    title.textContent = t(key);
    return title;
}

function renderDocumentTimeline(timeline, className = '') {
    const labels = {
        published: 'datePublished', modified: 'dateModified',
        indexed: 'dateIndexed', verified: 'dateVerified',
    };
    const list = document.createElement('dl');
    list.className = `document-timeline ${className}`.trim();
    list.setAttribute('aria-label', t('documentDates'));
    for (const [key, labelKey] of Object.entries(labels)) {
        const row = document.createElement('div');
        const term = document.createElement('dt');
        term.textContent = t(labelKey);
        const description = document.createElement('dd');
        const point = timeline?.[key] || {};
        if (point.value) {
            const time = document.createElement('time');
            time.dateTime = point.value;
            time.textContent = /^\d{4}-\d{2}-\d{2}$/.test(point.value)
                ? point.value
                : reviewDate(point.value);
            description.appendChild(time);
        } else {
            description.textContent = t('dateUnknown');
        }
        if (point.source) {
            const source = document.createElement('small');
            let sourceLabel = point.source;
            if (point.source.startsWith('document:')) {
                sourceLabel = t('dateSourceDocument', {
                    field: point.source.split(':').slice(2).join(':'),
                });
            } else if (point.source === 'filesystem:mtime') {
                sourceLabel = t('dateSourceFilesystem');
            } else if (point.source === 'grimore:index') {
                sourceLabel = t('dateSourceIndex');
            } else if (point.source.startsWith('human:')) {
                sourceLabel = t('dateSourceHuman');
            } else if (point.source.startsWith('legacy:')) {
                sourceLabel = t('dateSourceLegacy');
            }
            source.textContent = t('dateSource', { value: sourceLabel });
            description.appendChild(source);
        }
        row.append(term, description);
        list.appendChild(row);
    }
    return list;
}

function renderReviewInbox() {
    const data = state.reviewInbox;
    if (!data) return;
    const counts = data.counts || {};
    $('#review-count-contradiction').textContent = String(counts.contradiction || 0);
    $('#review-count-duplicate-exact').textContent = String(counts.duplicate_exact || 0);
    $('#review-count-duplicate-near').textContent = String(counts.duplicate_near || 0);
    $('#review-count-due').textContent = String(counts.review_due || 0);

    const filter = state.reviewFilter || 'all';
    $('#review-filter').value = filter;
    const items = (data.items || []).filter(item => filter === 'all' || item.kind === filter);
    const list = $('#review-items');
    list.innerHTML = '';

    if (items.length === 0) {
        const empty = document.createElement('li');
        empty.className = 'review-items__empty';
        empty.textContent = t('reviewEmpty');
        list.appendChild(empty);
    } else {
        for (const item of items) {
            const row = document.createElement('li');
            row.className = `review-item review-item--${item.kind}`;

            const head = document.createElement('div');
            head.className = 'review-item__head';
            const kind = document.createElement('span');
            kind.className = 'review-item__kind';
            kind.textContent = t(REVIEW_KIND_LABELS[item.kind] || item.kind);
            const status = document.createElement('span');
            status.className = 'review-item__status';
            status.textContent = t('candidate');
            head.append(kind, status);

            const titles = (item.documents || []).map(document => document.title || document.path);
            const title = document.createElement('strong');
            title.textContent = titles.join(' ↔ ') || t('documents');
            const reason = document.createElement('p');
            reason.className = 'review-item__reason';
            reason.textContent = `${t('reasonLabel')}: ${reviewReason(item)}`;
            const paths = document.createElement('small');
            paths.textContent = (item.documents || [])
                .map(document => document.path)
                .filter(Boolean)
                .join(' · ');
            row.append(head, title, reason);
            if (paths.textContent) row.appendChild(paths);

            const detail = document.createElement('details');
            detail.className = 'review-item__details';
            const summary = document.createElement('summary');
            summary.textContent = t('inspectCandidate');
            detail.appendChild(summary);

            const documents = document.createElement('section');
            documents.appendChild(reviewSectionTitle('involvedDocuments'));
            const documentList = document.createElement('ul');
            documentList.className = 'review-documents';
            for (const candidateDocument of item.documents || []) {
                const documentItem = document.createElement('li');
                const canOpen = candidateDocument.note_id != null;
                const documentTitle = canOpen
                    ? document.createElement('button')
                    : document.createElement('strong');
                if (canOpen) {
                    documentTitle.type = 'button';
                    documentTitle.addEventListener('click', () => openNote(candidateDocument.note_id));
                    documentTitle.title = t('openDocument');
                }
                documentTitle.textContent = candidateDocument.title || candidateDocument.path;
                const documentPath = document.createElement('small');
                documentPath.textContent = candidateDocument.path || '';
                documentItem.append(documentTitle, documentPath);
                if (candidateDocument.timeline) {
                    documentItem.appendChild(renderDocumentTimeline(
                        candidateDocument.timeline, 'document-timeline--compact',
                    ));
                }
                if (canOpen) {
                    const classification = candidateDocument.classification || {};
                    const classificationForm = document.createElement('div');
                    classificationForm.className = 'review-classification';
                    const classificationHead = document.createElement('div');
                    classificationHead.className = 'review-classification__head';
                    const classificationTitle = document.createElement('strong');
                    classificationTitle.textContent = t('classificationTitle');
                    const classificationState = document.createElement('span');
                    classificationState.className = classification.manual_override
                        ? 'review-classification__state is-manual'
                        : 'review-classification__state';
                    classificationState.textContent = t(
                        classification.manual_override ? 'classificationManual' : 'classificationIndexed',
                    );
                    classificationHead.append(classificationTitle, classificationState);

                    const fields = document.createElement('div');
                    fields.className = 'review-classification__fields';
                    const categoryLabel = document.createElement('label');
                    const categoryText = document.createElement('span');
                    categoryText.textContent = t('classificationCategory');
                    const categoryInput = document.createElement('input');
                    categoryInput.type = 'text';
                    categoryInput.maxLength = 200;
                    categoryInput.value = classification.category || '';
                    categoryLabel.append(categoryText, categoryInput);

                    const tagsLabel = document.createElement('label');
                    const tagsText = document.createElement('span');
                    tagsText.textContent = t('classificationTags');
                    const tagsInput = document.createElement('input');
                    tagsInput.type = 'text';
                    tagsInput.maxLength = 5100;
                    tagsInput.placeholder = t('classificationTagsHelp');
                    tagsInput.value = (classification.tags || []).join(', ');
                    tagsLabel.append(tagsText, tagsInput);
                    fields.append(categoryLabel, tagsLabel);

                    const classificationFooter = document.createElement('div');
                    classificationFooter.className = 'review-classification__footer';
                    const scope = document.createElement('small');
                    scope.textContent = t('classificationScope');
                    const save = document.createElement('button');
                    save.type = 'button';
                    save.className = 'classification-action';
                    save.textContent = t('classificationSave');
                    save.addEventListener('click', () => submitClassification(
                        candidateDocument.note_id, categoryInput.value, tagsInput.value,
                    ));
                    classificationFooter.append(scope, save);
                    classificationForm.append(classificationHead, fields, classificationFooter);
                    documentItem.appendChild(classificationForm);
                }
                documentList.appendChild(documentItem);
            }
            documents.appendChild(documentList);
            detail.appendChild(documents);

            const passageSection = document.createElement('section');
            passageSection.appendChild(reviewSectionTitle('candidatePassages'));
            const passageList = document.createElement('ol');
            passageList.className = 'review-passages';
            if (!item.passages || item.passages.length === 0) {
                const unavailable = document.createElement('li');
                unavailable.className = 'review-passages__empty';
                unavailable.textContent = t('noCandidatePassages');
                passageList.appendChild(unavailable);
            } else {
                for (const passage of item.passages) {
                    const passageItem = document.createElement('li');
                    const passageTitle = document.createElement('strong');
                    passageTitle.textContent = passage.title || passage.path || t('source');
                    const locator = document.createElement('small');
                    locator.textContent = reviewPassageLocator(passage);
                    const quote = document.createElement('blockquote');
                    quote.textContent = passage.text || '';
                    passageItem.append(passageTitle, locator, quote);
                    passageList.appendChild(passageItem);
                }
            }
            passageSection.appendChild(passageList);
            detail.appendChild(passageSection);

            const coverage = item.coverage || {};
            const coverageSection = document.createElement('section');
            coverageSection.className = 'review-coverage';
            coverageSection.appendChild(reviewSectionTitle('analysisCoverage'));
            const coverageLead = document.createElement('p');
            coverageLead.className = `review-coverage__status review-coverage__status--${coverage.status || 'unavailable'}`;
            const coverageBasis = REVIEW_COVERAGE_BASIS[coverage.basis];
            coverageLead.textContent = coverageBasis
                ? `${reviewCoverageStatus(coverage.status)} · ${t(coverageBasis)}`
                : reviewCoverageStatus(coverage.status);
            coverageSection.appendChild(coverageLead);
            const metrics = document.createElement('ul');
            metrics.className = 'review-coverage__metrics';
            const documentMetric = document.createElement('li');
            documentMetric.textContent = t('coverageDocuments', {
                analyzed: coverage.documents?.analyzed ?? 0,
                total: coverage.documents?.total ?? 0,
            });
            const passageMetric = document.createElement('li');
            passageMetric.textContent = t('coveragePassages', {
                shown: coverage.passages?.shown ?? 0,
                available: coverage.passages?.available ?? 0,
            });
            metrics.append(documentMetric, passageMetric);
            if (Number.isInteger(coverage.chunks_analyzed)) {
                const chunkMetric = document.createElement('li');
                chunkMetric.textContent = t('coverageChunks', { count: coverage.chunks_analyzed });
                metrics.appendChild(chunkMetric);
            }
            coverageSection.appendChild(metrics);
            if (coverage.caps?.max_claims_per_document) {
                const cap = document.createElement('p');
                cap.className = 'review-coverage__cap';
                cap.textContent = t('coverageClaimCap', {
                    sections: coverage.caps.max_sections_per_document,
                    characters: coverage.caps.max_characters_per_document,
                    claims: coverage.caps.max_claims_per_document,
                });
                coverageSection.appendChild(cap);
            }
            if (coverage.caps?.max_results) {
                const cap = document.createElement('p');
                cap.className = 'review-coverage__cap';
                cap.textContent = t('coverageResultCap', { count: coverage.caps.max_results });
                coverageSection.appendChild(cap);
            }
            const limitations = document.createElement('ul');
            limitations.className = 'review-coverage__limitations';
            for (const limitation of coverage.limitations || []) {
                const limitationItem = document.createElement('li');
                limitationItem.textContent = t(REVIEW_LIMITATIONS[limitation] || limitation);
                limitations.appendChild(limitationItem);
            }
            coverageSection.appendChild(limitations);
            detail.appendChild(coverageSection);

            const actions = document.createElement('section');
            actions.className = 'review-actions';
            actions.appendChild(reviewSectionTitle('reviewActions'));
            const actionRow = document.createElement('div');
            const confirm = document.createElement('button');
            confirm.type = 'button';
            confirm.className = 'review-action review-action--confirm';
            confirm.textContent = t('confirmCandidate');
            confirm.addEventListener('click', () => submitReviewDecision(item.id, 'confirm'));
            const dismiss = document.createElement('button');
            dismiss.type = 'button';
            dismiss.className = 'review-action review-action--dismiss';
            dismiss.textContent = t('dismissCandidate');
            dismiss.addEventListener('click', () => submitReviewDecision(item.id, 'dismiss'));
            const postponeLabel = document.createElement('label');
            const postponeText = document.createElement('span');
            postponeText.textContent = t('postponeFor');
            const postponeDays = document.createElement('select');
            postponeDays.className = 'review-action review-postpone-days';
            for (const [days, key] of [[7, 'days7'], [30, 'days30'], [90, 'days90']]) {
                const option = document.createElement('option');
                option.value = String(days);
                option.textContent = t(key);
                if (days === 30) option.selected = true;
                postponeDays.appendChild(option);
            }
            postponeLabel.append(postponeText, postponeDays);
            const postpone = document.createElement('button');
            postpone.type = 'button';
            postpone.className = 'review-action review-action--postpone';
            postpone.textContent = t('postponeCandidate');
            postpone.addEventListener('click', () => submitReviewDecision(
                item.id, 'postpone', Number(postponeDays.value),
            ));
            actionRow.append(confirm, dismiss, postponeLabel, postpone);
            actions.appendChild(actionRow);
            detail.appendChild(actions);
            row.appendChild(detail);
            list.appendChild(row);
        }
    }

    $('#review-status').textContent = t('reviewShowing', {
        count: items.length,
        suffix: items.length === 1 ? '' : 's',
    });
    renderReviewHistory(data.history || []);
    updateProtectedControls();
}

function reviewDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value || '';
    return date.toLocaleString(state.language, { dateStyle: 'medium', timeStyle: 'short' });
}

function renderReviewHistory(history) {
    const list = $('#review-history-items');
    list.innerHTML = '';
    if (!history.length) {
        const empty = document.createElement('li');
        empty.className = 'review-history__empty';
        empty.textContent = t('historyEmpty');
        list.appendChild(empty);
        return;
    }
    const decisionKeys = {
        confirm: 'decisionConfirm',
        dismiss: 'decisionDismiss',
        postpone: 'decisionPostpone',
    };
    for (const event of history) {
        const item = document.createElement('li');
        const documents = event.snapshot?.documents || [];
        const title = document.createElement('strong');
        title.textContent = documents
            .map(document => document.title || document.path)
            .filter(Boolean)
            .join(' ↔ ') || event.candidate_id;
        const decision = document.createElement('span');
        decision.className = `review-history__decision review-history__decision--${event.decision}`;
        decision.textContent = t(decisionKeys[event.decision] || event.decision);
        const when = document.createElement('small');
        const parts = [t('decisionRecordedAt', { value: reviewDate(event.decided_at) })];
        if (event.postpone_until) {
            parts.push(t('postponedUntil', { value: reviewDate(event.postpone_until) }));
        }
        when.textContent = parts.join(' · ');
        item.append(title, decision, when);
        list.appendChild(item);
    }
}

async function submitReviewDecision(candidateId, decision, postponeDays = null) {
    if (!state.writeEnabled || !state.writeToken || state.reviewDecisionBusy || state.reviewClassificationBusy) {
        $('#review-operation-status').dataset.guidance = 'true';
        updateProtectedControls();
        return;
    }
    state.reviewDecisionBusy = true;
    updateProtectedControls();
    const status = $('#review-operation-status');
    status.dataset.guidance = 'false';
    status.textContent = t('reviewDecisionSaving');
    const body = { decision };
    if (decision === 'postpone') body.postpone_days = postponeDays;
    try {
        const response = await apiFetch(
            `/api/review-inbox/${encodeURIComponent(candidateId)}/decision`,
            {
                method: 'POST',
                headers: protectedHeaders(),
                body: JSON.stringify(body),
            },
        );
        if (!response.ok) throw Object.assign(
            new Error(await responseError(response)), { status: response.status },
        );
        status.textContent = t('reviewDecisionSaved');
        await loadReviewInbox();
    } catch (error) {
        status.textContent = t('reviewDecisionFailed', {
            detail: protectedError(error, error.status),
        });
    } finally {
        state.reviewDecisionBusy = false;
        updateProtectedControls();
    }
}

async function submitClassification(noteId, categoryValue, tagsValue) {
    if (!state.writeEnabled || !state.writeToken || state.reviewDecisionBusy || state.reviewClassificationBusy) {
        $('#review-operation-status').dataset.guidance = 'true';
        updateProtectedControls();
        return;
    }
    state.reviewClassificationBusy = true;
    updateProtectedControls();
    const status = $('#review-operation-status');
    status.dataset.guidance = 'false';
    status.textContent = t('classificationSaving');
    const tags = tagsValue.split(',').map(tag => tag.trim()).filter(Boolean);
    try {
        const response = await apiFetch(`/api/notes/${encodeURIComponent(noteId)}/classification`, {
            method: 'POST',
            headers: protectedHeaders(),
            body: JSON.stringify({
                category: categoryValue.trim() || null,
                tags,
            }),
        });
        if (!response.ok) throw Object.assign(
            new Error(await responseError(response)), { status: response.status },
        );
        status.textContent = t('classificationSaved');
        await Promise.all([loadReviewInbox(), loadCategories(), loadFilterOptions()]);
    } catch (error) {
        status.textContent = t('classificationFailed', {
            detail: protectedError(error, error.status),
        });
    } finally {
        state.reviewClassificationBusy = false;
        updateProtectedControls();
    }
}

async function loadReviewInbox() {
    const button = $('#review-refresh');
    const status = $('#review-status');
    button.disabled = true;
    status.textContent = t('reviewLoading');
    try {
        const response = await apiFetch('/api/review-inbox', { cache: 'no-store' });
        if (!response.ok) throw new Error(await responseError(response));
        state.reviewInbox = await response.json();
        renderReviewInbox();
    } catch (error) {
        status.textContent = t('reviewUnavailable', { detail: friendlyError(error) });
    } finally {
        button.disabled = false;
    }
}

// ── Reviewable topic synthesis ─────────────────────────────────

const SYNTHESIS_LIMIT_LABELS = {
    generated_notes_excluded: 'synthesisLimitGenerated',
    missing_indexed_passages: 'synthesisLimitMissingPassages',
    context_limit: 'synthesisLimitContext',
    representative_passages_only: 'synthesisLimitRepresentative',
    unknown_citations: 'synthesisLimitUnknownCitations',
    sources_not_cited: 'synthesisLimitUncitedSources',
    no_matching_notes: 'synthesisLimitNoMatches',
};

function synthesisCoverageLabel(status) {
    return t({
        complete: 'synthesisCoverageComplete',
        partial: 'synthesisCoveragePartial',
        unavailable: 'synthesisCoverageUnavailable',
    }[status] || 'synthesisCoverageUnavailable');
}

function renderSynthesisPreview() {
    const report = state.synthesisPreview;
    if (!report) return;
    const preview = $('#synthesis-preview');
    preview.hidden = false;
    const coverage = report.coverage || {};
    const coverageStatus = coverage.status || 'unavailable';
    const badge = $('#synthesis-coverage-badge');
    badge.className = `synthesis-preview__badge synthesis-preview__badge--${coverageStatus}`;
    badge.textContent = synthesisCoverageLabel(coverageStatus);
    $('#synthesis-preview-title').textContent = report.title || report.selector || t('topicSynthesis');

    const notes = coverage.notes || {};
    const passages = coverage.passages || {};
    const context = coverage.context || {};
    const metrics = $('#synthesis-metrics');
    metrics.innerHTML = '';
    const metricTexts = [
        t('synthesisNotesMetric', {
            included: notes.included_in_prompt || 0,
            eligible: notes.eligible || 0,
        }),
        t('synthesisPassagesMetric', {
            selected: passages.selected || 0,
            available: passages.available || 0,
        }),
        t('synthesisContextMetric', {
            used: context.characters || 0,
            limit: context.max_characters || 0,
        }),
    ];
    for (const value of metricTexts) {
        const item = document.createElement('li');
        item.textContent = value;
        metrics.appendChild(item);
    }

    const limitations = $('#synthesis-limitations');
    limitations.innerHTML = '';
    for (const limitation of coverage.limitations || []) {
        const item = document.createElement('li');
        item.textContent = t(SYNTHESIS_LIMIT_LABELS[limitation] || limitation);
        limitations.appendChild(item);
    }
    limitations.hidden = limitations.children.length === 0;

    const body = $('#synthesis-preview-body');
    body.innerHTML = report.body ? renderCitedHtml(report.body) : '';
    body.hidden = !report.body;
    const sourcesByTitle = new Map(
        (report.source_details || []).map(source => [source.title, source]),
    );
    $$('.cite', body).forEach(citation => {
        const title = citation.dataset.label.split('#', 1)[0].trim();
        const source = sourcesByTitle.get(title);
        if (source?.note_id != null) {
            citation.title = t('synthesisOpenSource');
            makeKeyboardActivatable(citation, () => openNote(source.note_id), 'link');
        }
    });

    const sourceItems = $('#synthesis-source-items');
    sourceItems.innerHTML = '';
    if (!(report.source_details || []).length) {
        const empty = document.createElement('li');
        empty.className = 'synthesis-sources__empty';
        empty.textContent = t('synthesisNoSources');
        sourceItems.appendChild(empty);
    }
    for (const source of report.source_details || []) {
        const item = document.createElement('li');
        item.className = source.included_in_prompt
            ? 'synthesis-source synthesis-source--included'
            : 'synthesis-source synthesis-source--omitted';
        const head = document.createElement('div');
        head.className = 'synthesis-source__head';
        const open = document.createElement('button');
        open.type = 'button';
        open.textContent = source.title || source.path;
        open.title = t('synthesisOpenSource');
        if (source.note_id == null) {
            open.disabled = true;
        } else {
            open.addEventListener('click', () => openNote(source.note_id));
        }
        const stateLabel = document.createElement('span');
        const sourceStateKey = source.included_in_prompt
            ? 'synthesisSourceIncluded'
            : source.omission_reason === 'missing_indexed_passages'
                ? 'synthesisSourceUnavailable'
                : 'synthesisSourceOmitted';
        stateLabel.textContent = t(sourceStateKey);
        head.append(open, stateLabel);
        const meta = document.createElement('small');
        meta.textContent = `${source.path || ''} · ${t('synthesisSourceMetric', {
            selected: source.passages_selected || 0,
            available: source.passages_available || 0,
        })}`;
        item.append(head, meta);
        for (const passage of source.passages || []) {
            const quote = document.createElement('blockquote');
            quote.textContent = passage;
            item.appendChild(quote);
        }
        sourceItems.appendChild(item);
    }
}

async function previewTopicSynthesis(event) {
    event.preventDefault();
    if (state.synthesisBusy) return;
    const value = $('#synthesis-value').value.trim();
    const status = $('#synthesis-status');
    if (!value) {
        status.textContent = t('synthesisIdle');
        $('#synthesis-value').focus();
        return;
    }
    state.synthesisBusy = true;
    $('#synthesis-generate').disabled = true;
    status.textContent = t('synthesisLoading');
    try {
        const response = await apiFetch('/api/syntheses/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                kind: $('#synthesis-kind').value,
                value,
                passages_per_note: Number($('#synthesis-passages').value),
            }),
        });
        if (!response.ok) throw new Error(await responseError(response));
        state.synthesisPreview = await response.json();
        renderSynthesisPreview();
        status.textContent = state.synthesisPreview.body
            ? t('synthesisReady')
            : t('synthesisNoOutput', {
                detail: state.synthesisPreview.skipped_reason || t('synthesisNoReason'),
            });
    } catch (error) {
        status.textContent = t('synthesisFailed', { detail: friendlyError(error) });
    } finally {
        state.synthesisBusy = false;
        $('#synthesis-generate').disabled = false;
    }
}

// ── Conversation memory ───────────────────────────────────────

function conversationHistory() {
    return state.conversation.slice(-CONVERSATION_CONTEXT_TURNS).map(turn => ({
        q: turn.q,
        a: turn.a,
        sources: [...(turn.sources || [])],
    }));
}

function renderConversation() {
    const turns = $('#conversation-turns');
    turns.innerHTML = '';
    const history = conversationHistory();
    if (history.length === 0) {
        const empty = document.createElement('li');
        empty.className = 'conversation__empty';
        empty.textContent = t('conversationHelp');
        turns.appendChild(empty);
        $('#conversation-status').textContent = t('noConversation');
        $('#conversation-reset').disabled = true;
        return;
    }

    for (const turn of history) {
        const item = document.createElement('li');
        const question = document.createElement('strong');
        question.textContent = turn.q;
        const answer = document.createElement('p');
        answer.textContent = turn.a.length > 180
            ? `${turn.a.slice(0, 180)}…`
            : turn.a;
        const sources = document.createElement('small');
        const sourceCount = (turn.sources || []).length;
        sources.textContent = t('citedCount', {
            count: sourceCount,
            suffix: sourceCount === 1 ? '' : (state.language === 'es' ? 's' : 's'),
        });
        item.append(question, answer, sources);
        turns.appendChild(item);
    }
    $('#conversation-status').textContent = t('contextTurns', {
        count: history.length, max: CONVERSATION_CONTEXT_TURNS,
    });
    $('#conversation-reset').disabled = false;
}

function recordConversationTurn(question, answer, sources) {
    state.conversation.push({ q: question, a: answer, sources: sources || [] });
    state.conversation = state.conversation.slice(-CONVERSATION_CONTEXT_TURNS);
    renderConversation();
}

function resetConversation() {
    closeDrawer();
    state.conversation = [];
    state.lastSearchHits = [];
    state.sourceDetails = [];
    state.citationSelections = null;
    state.sourceLabels = [];
    state.retrievalScope = null;
    state.evidenceStatus = null;
    renderConversation();
    $('#answer').hidden = true;
    $('#answer-body').textContent = '';
    $('#sources').hidden = true;
    $('#sources-list').innerHTML = '';
    $('#dropped').hidden = true;
    $('#evidence-status').hidden = true;
    const results = $('#results');
    results.innerHTML = '';
    const empty = document.createElement('li');
    empty.className = 'hits__empty';
    empty.textContent = t('noPassagesYet');
    results.appendChild(empty);
    renderMarginalia([]);
    $('#question').value = '';
    hideRequestError();
    $('#question').focus();
}

// ── Retrieval scope ───────────────────────────────────────────

function selectedFilters() {
    const category = $('#filter-category').value.trim();
    const tags = [...new Set(
        $('#filter-tags').value.split(',').map(value => value.trim()).filter(Boolean),
    )];
    const formats = $$('input[type="checkbox"]:checked', $('#filter-formats'))
        .map(input => input.value);
    const filters = {};
    if (category) filters.category = category;
    if (tags.length) filters.tags = tags;
    if (formats.length) filters.formats = formats;
    return filters;
}

function updateFilterSummary() {
    const filters = selectedFilters();
    const parts = [];
    if (filters.category) parts.push(t('scopeCategory', { value: filters.category }));
    if (filters.tags) parts.push(t('scopeTags', { value: filters.tags.join(' + ') }));
    if (filters.formats) parts.push(t('scopeFormats', { value: filters.formats.join(' / ') }));
    $('#filter-summary').textContent = parts.length
        ? parts.join(' · ')
        : t('allDocuments');
    $('#filter-panel').classList.toggle('has-filters', parts.length > 0);
    $('#filter-status').textContent = parts.length
        ? t('scopeApplied')
        : '';
}

function setFilterCategory(category) {
    const select = $('#filter-category');
    if (![...select.options].some(option => option.value === category)) {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        select.appendChild(option);
    }
    select.value = category;
    $('#filter-panel').open = true;
    updateFilterSummary();
}

function renderFilterOptions(data) {
    state.filterOptions = data;
    const categorySelect = $('#filter-category');
    const selectedCategory = categorySelect.value;
    categorySelect.innerHTML = '';
    const allCategories = document.createElement('option');
    allCategories.value = '';
    allCategories.textContent = t('allCategories');
    categorySelect.appendChild(allCategories);
    for (const row of data.categories || []) {
        const option = document.createElement('option');
        option.value = row.category;
        option.textContent = `${row.category} (${row.count})`;
        categorySelect.appendChild(option);
    }
    if ([...categorySelect.options].some(option => option.value === selectedCategory)) {
        categorySelect.value = selectedCategory;
    }

    const tagOptions = $('#filter-tag-options');
    tagOptions.innerHTML = '';
    for (const row of data.tags || []) {
        const option = document.createElement('option');
        option.value = row.tag;
        option.label = state.language === 'es'
            ? `${row.count} documento${row.count === 1 ? '' : 's'}`
            : `${row.count} document${row.count === 1 ? '' : 's'}`;
        tagOptions.appendChild(option);
    }

    const formats = $('#filter-formats');
    formats.innerHTML = '';
    if (!data.formats || data.formats.length === 0) {
        const empty = document.createElement('span');
        empty.className = 'invocation-filter__empty';
        empty.textContent = t('noFormats');
        formats.appendChild(empty);
    } else {
        for (const [index, format] of data.formats.entries()) {
            const label = document.createElement('label');
            const input = document.createElement('input');
            input.type = 'checkbox';
            input.id = `filter-format-${index}`;
            input.value = format;
            input.addEventListener('change', updateFilterSummary);
            const text = document.createElement('span');
            text.textContent = format.toUpperCase();
            label.append(input, text);
            formats.appendChild(label);
        }
    }
    updateFilterSummary();
}

async function loadFilterOptions() {
    try {
        const response = await apiFetch('/api/filter-options', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderFilterOptions(await response.json());
    } catch (error) {
        $('#filter-status').textContent = t('scopeUnavailable', { detail: error.message });
        $('#filter-formats').textContent = t('formatsUnavailable');
    }
}

function clearFilters() {
    $('#filter-category').value = '';
    $('#filter-tags').value = '';
    $$('input[type="checkbox"]', $('#filter-formats')).forEach(input => {
        input.checked = false;
    });
    updateFilterSummary();
}

// ── Categories (the verso catalogue) ────────────────────────────

async function loadCategories() {
    try {
        const r = await apiFetch('/api/categories');
        const data = await r.json();
        state.categories = data.categories || [];
        renderCategories();
    } catch (e) {
        // Non-fatal — the rest of the UI keeps working.
        const ul = $('#categories');
        ul.innerHTML = '';
        const item = document.createElement('li');
        item.className = 'categories__loading';
        item.textContent = t('catalogueUnavailable');
        ul.appendChild(item);
    }
}

function renderCategories() {
    const ul = $('#categories');
    ul.innerHTML = '';
    if (state.categories.length === 0) {
        const li = document.createElement('li');
        li.className = 'categories__loading';
        li.textContent = t('noCategories');
        ul.appendChild(li);
        return;
    }
    for (const row of state.categories) {
        const li = document.createElement('li');
        const name = document.createElement('span');
        name.textContent = row.category || t('uncategorised');
        const count = document.createElement('span');
        count.className = 'count';
        count.textContent = String(row.count);
        li.append(name, count);
        makeKeyboardActivatable(li, () => {
            const q = $('#question');
            q.value = t('categoryQuestion', { category: row.category });
            setFilterCategory(row.category);
            q.focus();
            q.setSelectionRange(q.value.length, q.value.length);
        });
        ul.appendChild(li);
    }
}

// ── Marginalia (right column glosses) ───────────────────────────

function renderMarginalia(hits) {
    const wrap = $('#marginalia');
    wrap.innerHTML = '';
    if (!hits || hits.length === 0) {
        wrap.innerHTML = `
            <div class="marginalia__placeholder">
                <span class="marginalia__symbol" aria-hidden="true">⚜</span>
                <p>${escapeHtml(t('marginalHelp'))}</p>
            </div>`;
        return;
    }
    const head = document.createElement('h3');
    head.className = 'rubric rubric--sm';
    head.textContent = t('marginalGloss');
    head.style.marginBottom = '24px';
    wrap.appendChild(head);

    hits.slice(0, 4).forEach((h, i) => {
        const div = document.createElement('div');
        div.className = 'marg-entry';
        div.innerHTML = `
            <div class="marg-entry__index">${escapeHtml(romanize(i + 1))} · ${escapeHtml(t('note'))} ${escapeHtml(String(h.note_id))}</div>
            <div class="marg-entry__title">${escapeHtml(h.title || `#${h.note_id}`)}</div>
            <div class="marg-entry__snippet">${escapeHtml(h.snippet || '')}</div>
            <span class="marg-entry__score">${escapeHtml(t('score'))} · ${Number(h.score || 0).toFixed(3)}</span>
        `;
        makeKeyboardActivatable(div, () => openNote(h.note_id));
        wrap.appendChild(div);
    });
}

// ── Hits (centre column, full list) ─────────────────────────────

function renderHits(hits, retrievalScope = null) {
    const wrap = $('#results');
    wrap.innerHTML = '';
    if (!hits || hits.length === 0) {
        const empty = document.createElement('li');
        empty.className = 'hits__empty';
        empty.textContent = retrievalScope?.status === 'no_matches'
            ? t('noScopeMatches')
            : t('noPassages');
        wrap.appendChild(empty);
        return;
    }
    for (const h of hits) {
        const li = document.createElement('li');
        li.className = 'hit';
        li.innerHTML = `
            <div class="hit__head">
                <div class="hit__title">${escapeHtml(h.title || `#${h.note_id}`)}</div>
                <div class="hit__score">${escapeHtml(t('score'))} · ${Number(h.score || 0).toFixed(3)}</div>
            </div>
            <div class="hit__snippet">${escapeHtml(h.snippet || '')}</div>
        `;
        makeKeyboardActivatable(li, () => openNote(h.note_id));
        wrap.appendChild(li);
    }
}

function renderEvidenceStatus(status, sourceDetails, retrievalScope = null) {
    const panel = $('#evidence-status');
    const count = (sourceDetails || []).length;
    const suffix = count === 1 ? '' : 's';
    const messages = {
        supported: count > 0
            ? t('evidenceSupported', { count, suffix })
            : t('evidenceUnavailable'),
        conflicting_evidence: t('evidenceConflict', { count }),
        uncertain: t('evidenceUncertain'),
        insufficient_evidence: retrievalScope?.status === 'no_matches'
            ? t('evidenceNoMatches')
            : t('evidenceInsufficient'),
        error: t('evidenceError'),
    };
    const pdfSources = new Map();
    for (const detail of sourceDetails || []) {
        const coverage = detail?.indexed_extraction;
        if (!coverage || !['partial', 'unknown'].includes(coverage.status)) continue;
        const key = detail?.document?.note_id ?? detail?.label;
        if (key == null) continue;
        const prior = pdfSources.get(key);
        if (prior !== 'partial' || coverage.status === 'partial') {
            pdfSources.set(key, coverage.status);
        }
    }
    const partialCount = [...pdfSources.values()].filter(value => value === 'partial').length;
    const unknownCount = [...pdfSources.values()].filter(value => value === 'unknown').length;
    const coverageWarnings = [];
    if (partialCount) coverageWarnings.push(t('evidencePdfPartial', { count: partialCount }));
    if (unknownCount) coverageWarnings.push(t('evidencePdfUnknown', { count: unknownCount }));
    panel.className = `response__evidence response__evidence--${status || 'uncertain'}`;
    if (coverageWarnings.length) panel.classList.add('response__evidence--coverage-warning');
    panel.textContent = [messages[status] || messages.uncertain, ...coverageWarnings].join(' ');
    panel.hidden = false;
}

// ── Note drawer ─────────────────────────────────────────────────

let drawerReturnFocus = null;

function closeDrawer({ restoreFocus = true } = {}) {
    const drawer = $('#drawer');
    const backdrop = $('#drawer-backdrop');
    if (drawer) drawer.remove();
    if (backdrop) backdrop.remove();
    document.removeEventListener('keydown', handleDrawerKeydown);
    if (restoreFocus && drawerReturnFocus?.isConnected) drawerReturnFocus.focus();
    if (restoreFocus) drawerReturnFocus = null;
}

function handleDrawerKeydown(event) {
    if (event.key === 'Escape') {
        event.preventDefault();
        closeDrawer();
        return;
    }
    if (event.key !== 'Tab') return;
    const drawer = $('#drawer');
    if (!drawer) return;
    const focusable = $$('button:not(:disabled), select:not(:disabled), [href], [tabindex]:not([tabindex="-1"])', drawer)
        .filter(element => !element.closest('[hidden]'));
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
    }
}

function activateDrawer(drawer, returnFocus = document.activeElement) {
    drawerReturnFocus = returnFocus;
    document.addEventListener('keydown', handleDrawerKeydown);
    $('.drawer__close', drawer)?.focus();
}

const SOURCE_STATE_MESSAGES = {
    current: 'sourceCurrent', modified: 'sourceModified', missing: 'sourceMissing',
    unavailable: 'sourceUnavailable', unverified: 'sourceUnverified', provided: 'sourceProvided',
};

function sourceDetailIndices(label, details = state.sourceDetails) {
    const title = String(label || '').split('#')[0].trim();
    return details.flatMap((detail, index) => {
        if (!detail || typeof detail.label !== 'string') return [];
        const matches = label.includes('#') ? detail.label === label
            : detail.label.split('#')[0].trim() === title;
        return matches ? [index] : [];
    });
}

function citationTarget(label, citationIndex = null, details = state.sourceDetails,
                        selections = state.citationSelections) {
    const indices = sourceDetailIndices(label, details);
    // The source list has no citation occurrence. Historical responses can
    // inspect a unique match, but never silently pick from shared labels.
    if (citationIndex === null || selections === null) {
        return { kind: indices.length === 1 ? 'legacy' : 'choose', indices,
                 reason: 'ambiguousCitation' };
    }
    const occurrences = Array.isArray(selections) ? selections.filter(selection =>
        selection && selection.citation_index === citationIndex,
    ) : [];
    const selection = occurrences.length === 1 && occurrences[0].label === label ? occurrences[0] : null;
    const chosen = selection?.source_detail_indices;
    if (Number.isInteger(citationIndex) && citationIndex >= 0 && selection?.selection === 'exact'
        && Array.isArray(chosen) && chosen.length === 1 && Number.isInteger(chosen[0])
        && indices.includes(chosen[0])) {
        const index = chosen[0];
        if (selection.form === 'short' && /^cite:[1-9]\d*$/.test(selection.citation_ref || '')) {
            const refs = details.flatMap((detail, i) =>
                detail?.citation_ref === selection.citation_ref ? [i] : [],
            );
            if (refs.length === 1 && refs[0] === index && details[index].label === label) {
                return { kind: 'exact', indices: [index] };
            }
        } else if (selection.form === 'wikilink' && selection.citation_ref == null
                   && indices.length === 1) {
            return { kind: 'exact', indices: [index] };
        }
    }
    return { kind: 'choose', indices,
             reason: selection?.selection === 'ambiguous' ? 'ambiguousCitation' : 'unverifiableSelection' };
}

function openSourceChoices(label, indices, reason = 'ambiguousCitation') {
    const returnFocus = drawerReturnFocus || document.activeElement;
    closeDrawer({ restoreFocus: false });
    const backdrop = document.createElement('div');
    backdrop.id = 'drawer-backdrop';
    backdrop.className = 'drawer-backdrop';
    backdrop.addEventListener('click', closeDrawer);
    const drawer = document.createElement('aside');
    drawer.id = 'drawer';
    drawer.className = 'drawer source-viewer';
    drawer.setAttribute('role', 'dialog');
    drawer.setAttribute('aria-modal', 'true');
    drawer.setAttribute('aria-label', t('chooseCitationPassage'));
    drawer.innerHTML = `
        <button class="drawer__close" aria-label="${escapeHtml(t('close'))}">${escapeHtml(t('close'))} ✕</button>
        <div class="drawer__rubric">— ${escapeHtml(t('chooseCitationPassage'))} —</div>
        <h2 class="drawer__title"></h2>
        <p class="source-viewer__state" role="status"></p>
        <div class="source-viewer__choices"></div>
    `;
    $('.drawer__title', drawer).textContent = label;
    $('.source-viewer__state', drawer).textContent = t(reason);
    const choices = $('.source-viewer__choices', drawer);
    for (const index of indices) {
        const detail = state.sourceDetails[index];
        if (!detail) continue;
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'source-viewer__choice';
        const locator = detail.locator || {};
        const heading = document.createElement('strong');
        heading.textContent = [
            detail.document?.path || detail.label,
            locator.page != null ? t('page', { value: locator.page }) : null,
            locator.heading,
            locator.chunk_index != null ? t('passage', { value: Number(locator.chunk_index) + 1 }) : null,
        ].filter(Boolean).join(' · ');
        const passage = document.createElement('span');
        passage.textContent = String(detail.passage?.text || t('noIndexedPassage')).slice(0, 280);
        button.append(heading, passage);
        button.addEventListener('click', () => openSourceDetail(index));
        choices.appendChild(button);
    }
    document.body.append(backdrop, drawer);
    $('.drawer__close', drawer).addEventListener('click', closeDrawer);
    activateDrawer(drawer, returnFocus);
}

function appendHighlightedText(container, text, passage) {
    const content = String(text || '');
    const needle = String(passage || '');
    const offset = needle ? content.indexOf(needle) : -1;
    container.textContent = '';
    if (offset === -1) {
        container.textContent = content;
        return false;
    }
    container.appendChild(document.createTextNode(content.slice(0, offset)));
    const mark = document.createElement('mark');
    mark.id = 'source-highlight';
    mark.textContent = needle;
    container.appendChild(mark);
    container.appendChild(document.createTextNode(content.slice(offset + needle.length)));
    return true;
}

function renderNavigableDocument(container, note, passage, locator, jump) {
    const content = String(note.body || '');
    const sections = Array.isArray(note.sections) ? note.sections : [];
    if (sections.length === 0) {
        return appendHighlightedText(container, content, passage);
    }

    container.textContent = '';
    const select = $('select', jump);
    select.innerHTML = '';
    jump.hidden = false;
    let highlighted = false;
    let highlightedIndex = -1;

    for (const [index, section] of sections.entries()) {
        const start = Math.max(0, Number(section.start) || 0);
        const end = Math.min(content.length, Number(section.end) || content.length);
        const block = document.createElement('section');
        block.className = 'source-viewer__document-section';
        block.id = `document-section-${index}`;
        const heading = document.createElement('h4');
        const labels = [];
        if (section.page != null) labels.push(t('page', { value: section.page }));
        if (section.heading) labels.push(section.heading);
        heading.textContent = labels.join(' · ') || t('section', { value: index + 1 });
        const text = document.createElement('div');
        const found = !highlighted && appendHighlightedText(
            text, content.slice(start, end), passage,
        );
        if (found) {
            highlighted = true;
            highlightedIndex = index;
        }
        block.append(heading, text);
        container.appendChild(block);

        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = heading.textContent;
        select.appendChild(option);
    }

    let targetIndex = sections.findIndex(section =>
        locator.page != null && Number(section.page) === Number(locator.page),
    );
    if (targetIndex === -1 && locator.heading) {
        targetIndex = sections.findIndex(section => section.heading === locator.heading);
    }
    if (targetIndex === -1) targetIndex = highlightedIndex;
    if (targetIndex >= 0) select.value = String(targetIndex);
    select.addEventListener('change', () => {
        $(`#document-section-${select.value}`, container)?.scrollIntoView({
            behavior: 'smooth', block: 'start',
        });
    });
    return highlighted;
}

async function openSourceDetail(index) {
    const detail = state.sourceDetails[index];
    if (!detail) return;
    const returnFocus = drawerReturnFocus || document.activeElement;
    closeDrawer({ restoreFocus: false });

    const backdrop = document.createElement('div');
    backdrop.id = 'drawer-backdrop';
    backdrop.className = 'drawer-backdrop';
    backdrop.addEventListener('click', closeDrawer);
    document.body.appendChild(backdrop);

    const drawer = document.createElement('aside');
    drawer.id = 'drawer';
    drawer.className = 'drawer source-viewer';
    drawer.setAttribute('role', 'dialog');
    drawer.setAttribute('aria-modal', 'true');
    drawer.setAttribute('aria-label', t('sourceEvidence'));
    drawer.innerHTML = `
        <button class="drawer__close" aria-label="${escapeHtml(t('close'))}">${escapeHtml(t('close'))} ✕</button>
        <div class="drawer__rubric">— ${escapeHtml(t('sourceEvidence'))} —</div>
        <nav class="source-viewer__nav" aria-label="${escapeHtml(t('retrievedPassages'))}">
            <button type="button" data-source-nav="previous">← ${escapeHtml(t('previous'))}</button>
            <span></span>
            <button type="button" data-source-nav="next">${escapeHtml(t('next'))} →</button>
        </nav>
        <h2 class="drawer__title"></h2>
        <div class="drawer__path"></div>
        <div class="source-viewer__locator"></div>
        <div class="source-viewer__state" role="status"></div>
        <div class="source-viewer__timeline" hidden></div>
        <section class="source-viewer__section">
            <h3>${escapeHtml(t('indexedPassage'))}</h3>
            <blockquote class="source-viewer__passage"><mark></mark></blockquote>
        </section>
        <section class="source-viewer__section source-viewer__document" hidden>
            <h3>${escapeHtml(t('currentDocument'))}</h3>
            <label class="source-viewer__jump" hidden>
                <span>${escapeHtml(t('goTo'))}</span>
                <select></select>
            </label>
            <div class="drawer__body drawer__body--loading">${escapeHtml(t('locatingPassage'))}</div>
            <p class="source-viewer__notice" hidden></p>
        </section>
    `;
    document.body.appendChild(drawer);
    $('.drawer__close', drawer).addEventListener('click', closeDrawer);
    activateDrawer(drawer, returnFocus);

    const documentInfo = detail.document || {};
    const locator = detail.locator || {};
    const passage = detail.passage || {};
    $('.drawer__title', drawer).textContent = documentInfo.title || detail.label || t('source');
    $('.drawer__path', drawer).textContent = [
        documentInfo.path,
        documentInfo.format ? String(documentInfo.format).toUpperCase() : null,
    ].filter(Boolean).join(' · ') || t('noDocumentPath');
    $('.source-viewer__state', drawer).textContent =
        t(SOURCE_STATE_MESSAGES[detail.source_state] || SOURCE_STATE_MESSAGES.unverified);
    $('.source-viewer__state', drawer).classList.add(
        `source-viewer__state--${detail.source_state || 'unverified'}`,
    );
    $('.source-viewer__passage mark', drawer).textContent =
        passage.text || t('noIndexedPassage');

    const locators = [];
    if (locator.page != null) locators.push(t('page', { value: locator.page }));
    if (locator.heading) locators.push(t('section', { value: locator.heading }));
    if (locator.chunk_index != null) locators.push(t('passage', { value: Number(locator.chunk_index) + 1 }));
    if (passage.completeness === 'excerpt') locators.push(t('excerptOnly'));
    $('.source-viewer__locator', drawer).textContent =
        locators.join(' · ') || t('noLocator');

    const previous = $('[data-source-nav="previous"]', drawer);
    const next = $('[data-source-nav="next"]', drawer);
    previous.disabled = index <= 0;
    next.disabled = index >= state.sourceDetails.length - 1;
    $('.source-viewer__nav span', drawer).textContent =
        t('position', { current: index + 1, total: state.sourceDetails.length });
    previous.addEventListener('click', () => openSourceDetail(index - 1));
    next.addEventListener('click', () => openSourceDetail(index + 1));

    if (detail.source_state === 'current' && documentInfo.note_id != null) {
        const section = $('.source-viewer__document', drawer);
        const body = $('.drawer__body', section);
        const notice = $('.source-viewer__notice', section);
        section.hidden = false;
        try {
            const response = await apiFetch(
                `/api/notes/${encodeURIComponent(documentInfo.note_id)}`,
                { cache: 'no-store' },
            );
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const note = await response.json();
            const timelineHost = $('.source-viewer__timeline', drawer);
            timelineHost.hidden = false;
            timelineHost.replaceChildren(renderDocumentTimeline(
                note.timeline, 'document-timeline--source',
            ));
            body.classList.remove('drawer__body--loading');
            const extractionWarning = renderExtractionWarning(note.extraction);
            if (extractionWarning) body.before(extractionWarning);
            const highlighted = renderNavigableDocument(
                body,
                note,
                passage.text,
                locator,
                $('.source-viewer__jump', section),
            );
            if (highlighted) {
                $('#source-highlight', body)?.scrollIntoView({ block: 'center' });
            } else {
                notice.hidden = false;
                notice.textContent = t('passageNotLocated');
            }
        } catch (error) {
            body.classList.remove('drawer__body--loading');
            body.classList.add('drawer__body--error');
            body.textContent = t('documentOpenFailed', { detail: friendlyError(error) });
        }
    }
}

async function openNote(noteId) {
    const returnFocus = drawerReturnFocus || document.activeElement;
    closeDrawer({ restoreFocus: false });

    const backdrop = document.createElement('div');
    backdrop.id = 'drawer-backdrop';
    backdrop.className = 'drawer-backdrop';
    backdrop.addEventListener('click', closeDrawer);
    document.body.appendChild(backdrop);

    const drawer = document.createElement('aside');
    drawer.id = 'drawer';
    drawer.className = 'drawer';
    drawer.setAttribute('role', 'dialog');
    drawer.setAttribute('aria-label', t('note'));
    drawer.innerHTML = `
        <button class="drawer__close" aria-label="${escapeHtml(t('close'))}">${escapeHtml(t('close'))} ✕</button>
        <div class="drawer__rubric">— ${escapeHtml(t('folio'))} —</div>
        <h2 class="drawer__title">${escapeHtml(t('summoning'))}</h2>
        <div class="drawer__path"></div>
        <div class="drawer__body drawer__body--loading">${escapeHtml(t('consulting'))}</div>
    `;
    document.body.appendChild(drawer);
    $('.drawer__close', drawer).addEventListener('click', closeDrawer);
    activateDrawer(drawer, returnFocus);

    try {
        const r = await apiFetch(`/api/notes/${encodeURIComponent(noteId)}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        $('.drawer__title', drawer).textContent = data.title || `${t('note')} #${noteId}`;
        $('.drawer__path', drawer).textContent = data.path || '';
        const body = $('.drawer__body', drawer);
        body.before(renderDocumentTimeline(data.timeline, 'document-timeline--drawer'));
        const extractionWarning = renderExtractionWarning(data.extraction);
        if (extractionWarning) body.before(extractionWarning);
        body.classList.remove('drawer__body--loading');
        body.textContent = data.body || t('emptyFolio');
    } catch (e) {
        $('.drawer__title', drawer).textContent = t('folioResists');
        const body = $('.drawer__body', drawer);
        body.classList.remove('drawer__body--loading');
        body.classList.add('drawer__body--error');
        body.textContent = t('noteLoadFailed', { id: noteId, detail: friendlyError(e) });
    }
}

// Resolve a citation label ([[Title]] or [[Title#anchor]]) by searching
// for the bare title and opening the best match.
async function openNoteByLabel(label, citationIndex = null) {
    const target = citationTarget(label, citationIndex);
    if (target.indices.length) {
        if (target.kind === 'choose') openSourceChoices(label, target.indices, target.reason);
        else openSourceDetail(target.indices[0]);
        return;
    }
    // A trace was supplied but this occurrence cannot be bound to a delivered
    // record. Do not turn a search result into its supposed selected passage.
    if (citationIndex !== null && state.citationSelections !== null) {
        openSourceChoices(label, [], 'unverifiableSelection');
        return;
    }
    const title = label.split('#')[0].trim();
    // Prefer the cached search hits when one of them matches — avoids a
    // round trip when the citation came from this turn's retrieval.
    const cached = state.lastSearchHits.find(h => h.title === title);
    if (cached) {
        openNote(cached.note_id);
        return;
    }
    try {
        const r = await apiFetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: title, top_k: 5 }),
        });
        if (!r.ok) throw new Error(await responseError(r));
        const data = await r.json();
        const exact = (data.hits || []).find(h => h.title === title);
        const hit = exact || (data.hits || [])[0];
        if (hit) openNote(hit.note_id);
    } catch (e) {
        showRequestError(t('searchFailed', { detail: friendlyError(e) }));
    }
}

// Re-wire .cite spans after streaming/render completes.
function wireCitations() {
    const body = $('#answer-body');
    if (!body) return;
    $$('.cite', body).forEach(el => {
        const index = Number(el.dataset.citationIndex);
        const target = citationTarget(el.dataset.label, index);
        el.title = t(target.kind === 'exact' ? 'selectedCitation'
            : target.kind === 'choose' ? 'chooseCitation' : 'inspectEvidence');
        if (el.dataset.citationWired) return;
        el.dataset.citationWired = 'true';
        makeKeyboardActivatable(el, () => openNoteByLabel(el.dataset.label, index), 'link');
    });
}

// ── Ask flow ────────────────────────────────────────────────────

function friendlyError(error) {
    return error instanceof TypeError ? t('networkError') : (error?.message || String(error));
}

async function responseError(response) {
    try {
        const data = await response.clone().json();
        return data.detail || data.error || `HTTP ${response.status}`;
    } catch {
        return `HTTP ${response.status}`;
    }
}

function hideRequestError() {
    const panel = $('#request-error');
    panel.hidden = true;
    panel.textContent = '';
}

function showRequestError(message, { focus = true } = {}) {
    const panel = $('#request-error');
    panel.textContent = message;
    panel.hidden = false;
    if (focus) panel.focus();
}

async function handleAsk(event) {
    event.preventDefault();
    if (state.busy) return;

    const q = $('#question').value.trim();
    if (!q) {
        showRequestError(t('enterQuestion'));
        return;
    }
    hideRequestError();
    const topK = parseInt($('#top-k').value, 10) || 5;
    const filters = selectedFilters();
    const history = conversationHistory();

    state.busy = true;
    closeDrawer();
    const btn = $('#ask-btn');
    btn.disabled = true;
    $('#conversation-reset').disabled = true;
    const btnText = $('.invocation__btn-text', btn);
    btnText.textContent = t('requestWorking');

    // Reveal panels.
    const panel  = $('#answer');
    const body   = $('#answer-body');
    const sources = $('#sources');
    const sourcesList = $('#sources-list');
    const dropped = $('#dropped');

    panel.hidden = false;
    sources.hidden = true;
    sourcesList.innerHTML = '';
    dropped.hidden = true;
    dropped.textContent = '';
    $('#evidence-status').hidden = true;
    state.sourceDetails = [];
    state.citationSelections = null;
    state.sourceLabels = [];
    state.retrievalScope = null;
    state.evidenceStatus = null;

    // Initial: just a scribe cursor pulsing.
    const textNode = document.createElement('span');
    const cursorNode = document.createElement('span');
    cursorNode.className = 'scribe-cursor';
    body.innerHTML = '';
    body.append(textNode, cursorNode);

    // Fire the search in parallel — the marginalia + hits don't depend
    // on the answer.
    const searchPromise = apiFetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, top_k: 10, history, ...filters }),
    }).then(async response => {
        if (!response.ok) throw new Error(await responseError(response));
        return response.json();
    }).catch(error => ({ hits: [], requestError: friendlyError(error) }));

    let collected = '';
    let streamedOk = false;
    let terminalEvent = null;

    try {
        const r = await apiFetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: q, top_k: topK, stream: true, history, ...filters,
            }),
        });
        if (!r.ok) throw new Error(await responseError(r));
        if (!r.body) throw new Error(t('noStreamBody'));

        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // SSE frames are separated by blank lines.
            let frameEnd;
            while ((frameEnd = buffer.indexOf('\n\n')) !== -1) {
                const frame = buffer.slice(0, frameEnd);
                buffer = buffer.slice(frameEnd + 2);

                for (const line of frame.split('\n')) {
                    if (!line.startsWith('data: ')) continue;
                    const payload = line.slice(6);
                    let ev;
                    try { ev = JSON.parse(payload); } catch { continue; }

                    if (ev.type === 'token' && ev.text) {
                        collected += ev.text;
                        textNode.textContent = collected;
                    } else if (ev.type === 'done') {
                        terminalEvent = ev;
                        if (typeof ev.answer === 'string') {
                            collected = ev.answer;
                            textNode.textContent = collected;
                        }
                        state.sourceDetails = Array.isArray(ev.source_details)
                            ? ev.source_details
                            : [];
                        state.citationSelections = Array.isArray(ev.citation_selections)
                            ? ev.citation_selections : null;
                        state.sourceLabels = ev.sources || [];
                        state.retrievalScope = ev.retrieval_scope || null;
                        // A model that never answered says nothing about the
                        // evidence: report the failure, not "insufficient".
                        state.evidenceStatus = ev.generation_error
                            ? 'error' : (ev.answer_status || 'uncertain');
                        renderSources(state.sourceLabels, state.sourceDetails);
                        renderEvidenceStatus(
                            state.evidenceStatus,
                            state.sourceDetails,
                            state.retrievalScope,
                        );
                        const drop = ev.dropped_citations || 0;
                        if (ev.retrieval_scope?.status === 'no_matches') {
                            dropped.hidden = false;
                            dropped.textContent =
                                t('noMatchesHelp');
                        } else if (drop > 0) {
                            dropped.hidden = false;
                            dropped.textContent = t('citationsRemoved', {
                                count: drop, suffix: drop === 1 ? '' : 's',
                            });
                        }
                    }
                }
            }
            streamedOk = true;
        }

        // Final render: paragraph + citation styling, drop the cursor.
        if (collected.trim()) {
            body.innerHTML = renderProse(collected);
            wireCitations();
        } else if (streamedOk) {
            // Without a generation error, an empty final event means no
            // passage was relevant enough to ask the model at all.
            const key = terminalEvent && !terminalEvent.generation_error ? 'noPassages' : 'noTokens';
            body.innerHTML = `<p><em>${escapeHtml(t(key))}</em></p>`;
        }
    } catch (e) {
        body.innerHTML = `<p><em>${escapeHtml(t('answerFailed'))}</em></p>`;
        renderEvidenceStatus('error', []);
        state.evidenceStatus = 'error';
        showRequestError(t('requestFailed', { detail: friendlyError(e) }));
        console.error('ask failed:', e);
    }

    if (terminalEvent && collected.trim()) {
        recordConversationTurn(q, collected, terminalEvent.sources || []);
    }

    // Wait for the parallel search to complete and render its results.
    const search = await searchPromise;
    if (search.requestError && $('#request-error').hidden) {
        showRequestError(t('searchFailed', { detail: search.requestError }), { focus: false });
    }
    state.lastSearchHits = search.hits || [];
    if (!state.retrievalScope) state.retrievalScope = search.retrieval_scope || null;
    renderHits(state.lastSearchHits, state.retrievalScope);
    renderMarginalia(state.lastSearchHits);

    state.busy = false;
    btn.disabled = false;
    btnText.textContent = t('invoke');
    renderConversation();
}

function renderSources(sourceLabels, sourceDetails = []) {
    const wrap = $('#sources');
    const list = $('#sources-list');
    list.innerHTML = '';
    if (!sourceLabels || sourceLabels.length === 0) {
        wrap.hidden = true;
        return;
    }
    wrap.hidden = false;
    for (const s of sourceLabels) {
        const li = document.createElement('li');
        li.textContent = `[[${s}]]`;
        li.dataset.label = s;
        const indices = sourceDetailIndices(s, sourceDetails);
        if (indices.length) {
            li.title = t(indices.length === 1 ? 'inspectEvidence' : 'chooseCitation');
            const sourceState = indices.length === 1
                ? sourceDetails[indices[0]].source_state : 'unverified';
            li.className = `source-link source-link--${sourceState || 'unverified'}`;
        }
        makeKeyboardActivatable(li, () => openNoteByLabel(s), 'link');
        list.appendChild(li);
    }
}

// ── Boot ────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    applyLanguage(preferredLanguage(), { persist: false });
    refreshHealth();
    loadCategories();
    loadIndexing();
    loadFilterOptions();
    loadReviewInbox();

    $('#language-select').addEventListener('change', event => {
        applyLanguage(event.target.value);
    });

    $('#setup-toggle').addEventListener('click', openSetup);
    $('#setup-close').addEventListener('click', closeSetup);
    $('#setup-refresh').addEventListener('click', loadSetup);
    $('#setup-copy').addEventListener('click', copySetupSnippet);
    $('#setup-apply').addEventListener('click', applySetupConfig);
    $('#operations-token').addEventListener('input', event => {
        state.writeToken = event.target.value;
        $('#indexing-operation-status').dataset.guidance = 'true';
        updateProtectedControls();
    });
    for (const id of ['#setup-vault', '#setup-chat', '#setup-embedding']) {
        $(id).addEventListener('input', updateSetupSnippet);
        $(id).addEventListener('change', updateSetupSnippet);
    }

    $('#filter-category').addEventListener('change', updateFilterSummary);
    $('#filter-tags').addEventListener('input', updateFilterSummary);
    $('#filter-reset').addEventListener('click', clearFilters);
    $('#conversation-reset').addEventListener('click', resetConversation);
    $('#indexing-preview').addEventListener('click', () => startIndexing('preview'));
    $('#indexing-apply').addEventListener('click', () => startIndexing('apply'));
    $('#review-refresh').addEventListener('click', loadReviewInbox);
    $('#review-filter').addEventListener('change', event => {
        state.reviewFilter = event.target.value;
        renderReviewInbox();
    });
    $('#synthesis-form').addEventListener('submit', previewTopicSynthesis);

    const form = $('#ask-form');
    form.addEventListener('submit', handleAsk);

    // Ctrl/Cmd + Enter from the textarea submits.
    $('#question').addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            form.requestSubmit();
        }
    });

    // Refresh the candle every 30 s so the user sees if the backend dies.
    setInterval(refreshHealth, 30_000);
});
