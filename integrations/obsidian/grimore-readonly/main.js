/* Desktop-only, dependency-free Obsidian plugin. No eval, shell or HTTP client. */
"use strict";
const { Plugin, ItemView, PluginSettingTab, Setting, Modal, Notice, TFile, MarkdownView } = require("obsidian");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

const VIEW = "grimore-readonly-panel";
const CLIENT_VERSION = "0.1.1";
const PROTOCOL = "2024-11-05";
const MAX_FRAME = 8 * 1024 * 1024;
const REQUIRED_TOOLS = ["grimore_library_info", "grimore_search", "grimore_get_source", "grimore_review_inbox"];
const DEFAULTS = { executable: "", cwd: "", profile: "", timeoutSeconds: 120, allowRemoteEmbeddings: false };
const KINDS = { contradiction: "Contradicciones", duplicate_exact: "Duplicados exactos", duplicate_near: "Parecidos", review_due: "Por revisar" };

function fingerprint(root, platform = process.platform) {
  const canonical = platform === "win32" ? root.replace(/\//g, "\\").toLowerCase() : root;
  return crypto.createHash("sha256").update("grimore-vault-v1\0" + canonical, "utf8").digest("hex");
}

function validateSettings(settings) {
  if (!settings || typeof settings.executable !== "string" || !path.isAbsolute(settings.executable)
      || typeof settings.cwd !== "string" || !path.isAbsolute(settings.cwd)
      || /[\x00-\x1f]/.test(settings.executable + settings.cwd)) {
    throw new Error("Configura rutas absolutas al ejecutable de Grimore y al directorio de configuración.");
  }
  if (typeof settings.profile !== "string" || (settings.profile && !/^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$/.test(settings.profile))) {
    throw new Error("El perfil solo admite letras, números, guion y guion bajo (hasta 80 caracteres).");
  }
  if (!Number.isInteger(settings.timeoutSeconds) || settings.timeoutSeconds < 5 || settings.timeoutSeconds > 300) {
    throw new Error("El tiempo de espera debe ser un entero entre 5 y 300 segundos.");
  }
  if (typeof settings.allowRemoteEmbeddings !== "boolean") throw new Error("La autorización remota debe ser explícita.");
}

function safeRelativePath(value) {
  if (typeof value !== "string" || !value || /[\\#\x00-\x1f]/.test(value) || path.posix.isAbsolute(value)
      || /^[A-Za-z]:/.test(value) || value.split("/").some(part => !part || part === "." || part === "..")) {
    throw new Error("La fuente no tiene una ruta relativa segura dentro de esta bóveda.");
  }
  return value;
}

function containedFile(root, relative, realpath = fs.realpathSync.native) {
  const resolved = realpath(path.join(root, safeRelativePath(relative)));
  const remainder = path.relative(root, resolved);
  if (!remainder || remainder === ".." || remainder.startsWith(".." + path.sep) || path.isAbsolute(remainder)) {
    throw new Error("El archivo resuelto está fuera de esta bóveda.");
  }
  return resolved;
}

function hashFile(filename) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(filename);
    stream.on("data", chunk => hash.update(chunk));
    stream.on("error", () => reject(new Error("No se pudo comprobar el archivo actual.")));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

class MCPClient {
  constructor({ spawnProcess = spawn, realpath = fs.realpathSync.native, onState = () => {} } = {}) {
    this.spawnProcess = spawnProcess;
    this.realpath = realpath;
    this.onState = onState;
    this.child = null;
    this.pending = new Map();
    this.counter = 0;
    this.buffer = "";
    this.ready = false;
    this.connecting = null;
    this.info = null;
    this.settings = null;
  }

  connect(settings, root) {
    if (this.connecting) return this.connecting;
    if (this.child) return Promise.reject(new Error("Desconecta antes de volver a conectar."));
    const promise = this.start(settings, root);
    this.connecting = promise;
    promise.finally(() => { if (this.connecting === promise) this.connecting = null; }).catch(() => {});
    return promise;
  }

  async start(settings, root) {
    validateSettings(settings);
    this.settings = { ...settings };
    const expected = fingerprint(this.realpath(root));
    const args = settings.profile ? ["--profile", settings.profile, "mcp"] : ["mcp"];
    const env = { ...process.env };
    // A profile is explicit; never silently inherit a different library.
    delete env.GRIMORE_PROFILE;
    let child;
    try {
      child = this.spawnProcess(settings.executable, args, {
        cwd: settings.cwd, env, shell: false, windowsHide: true, stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (_) {
      throw new Error("No se pudo iniciar Grimore. Revisa las rutas y permisos del ejecutable.");
    }
    this.child = child;
    this.buffer = "";
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", chunk => { if (this.child === child) this.receive(chunk); });
    child.stdout.on("end", () => { if (this.child === child) this.close("El servidor cerró la conexión local."); });
    child.stderr.on("data", () => {}); // Drain without retaining passages, credentials or logs.
    child.stdin.on("error", () => { if (this.child === child) this.close("Falló la conexión local con Grimore."); });
    child.on("error", () => { if (this.child === child) this.close("No se pudo iniciar Grimore. Revisa el ejecutable."); });
    child.on("exit", () => { if (this.child === child) this.close("El servidor local se detuvo.", false); });
    this.onState();
    try {
      const hello = await this.request("initialize", {
        protocolVersion: PROTOCOL, capabilities: {}, clientInfo: { name: "grimore-obsidian-readonly", version: CLIENT_VERSION },
      }, 15000);
      if (hello?.serverInfo?.name !== "grimore" || hello.protocolVersion !== PROTOCOL) {
        throw new Error("El proceso no anuncia un servidor Grimore compatible.");
      }
      this.send({ jsonrpc: "2.0", method: "notifications/initialized" });
      const discovery = await this.request("tools/list", {}, 15000);
      if (!Array.isArray(discovery.tools) || !REQUIRED_TOOLS.every(name => discovery.tools.some(t => t?.name === name))) {
        throw new Error("Actualiza Grimore: faltan las consultas de la fase 5.");
      }
      const info = this.decodeTool(await this.request("tools/call", { name: "grimore_library_info", arguments: {} }, 15000));
      if (info?.contract_version !== 1 || info.vault_identity?.algorithm !== "sha256-local-path-v1"
          || info.vault_identity.fingerprint !== expected) {
        throw new Error("La biblioteca de Grimore no coincide con la bóveda abierta. No se consultará contenido.");
      }
      if (!["local", "remote"].includes(info.model_execution?.embeddings?.mode)) {
        throw new Error("Grimore no informa el destino de la búsqueda. No se enviará la consulta.");
      }
      if (this.child !== child) throw new Error("La conexión se cerró.");
      this.info = info;
      this.ready = true;
      this.onState();
      return info;
    } catch (error) {
      if (this.child === child) this.close();
      throw error;
    }
  }

  send(message) {
    if (!this.child) throw new Error("Conecta primero con Grimore.");
    this.child.stdin.write(JSON.stringify(message) + "\n");
  }

  request(method, params, timeoutMs = (this.settings?.timeoutSeconds || 120) * 1000) {
    if (!this.child) return Promise.reject(new Error("Conecta primero con Grimore."));
    if (this.pending.size) return Promise.reject(new Error("Espera a que termine la consulta en curso."));
    const id = ++this.counter;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.close("La consulta excedió el tiempo de espera. Se cerró la conexión; un servicio de modelos puede seguir trabajando."), timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try { this.send({ jsonrpc: "2.0", id, method, params }); }
      catch (_) { this.close("No se pudo enviar la consulta al proceso local."); }
    });
  }

  receive(chunk) {
    this.buffer += chunk;
    let newline;
    while ((newline = this.buffer.indexOf("\n")) !== -1) {
      const frame = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (!frame.trim()) continue;
      if (Buffer.byteLength(frame, "utf8") > MAX_FRAME) { this.close("La respuesta supera el límite de 8 MiB."); return; }
      let message;
      try { message = JSON.parse(frame); }
      catch (_) { this.close("El servidor emitió una respuesta no válida."); return; }
      if (!message || message.jsonrpc !== "2.0" || (message.id === undefined && typeof message.method !== "string")) {
        this.close("El servidor emitió una respuesta no válida."); return;
      }
      if (message.id === undefined) continue;
      if (!Number.isSafeInteger(message.id) || message.id < 1) {
        this.close("El servidor emitió un identificador de respuesta no válido."); return;
      }
      const pending = this.pending.get(message.id);
      if (!pending) continue;
      this.pending.delete(message.id);
      clearTimeout(pending.timer);
      if (message.error) pending.reject(new Error("Grimore rechazó la consulta. Revisa la configuración y los campos solicitados."));
      else if (!Object.hasOwn(message, "result")) {
        pending.reject(new Error("La respuesta de Grimore está incompleta."));
        this.close();
      } else pending.resolve(message.result);
    }
    if (Buffer.byteLength(this.buffer, "utf8") > MAX_FRAME) this.close("La respuesta supera el límite de 8 MiB.");
  }

  decodeTool(result) {
    if (result?.isError) throw new Error("La consulta falló en Grimore. No se ha modificado ninguna nota.");
    if (!Array.isArray(result?.content) || result.content.length !== 1 || result.content[0]?.type !== "text") {
      throw new Error("La respuesta de la herramienta no tiene el formato esperado.");
    }
    try { return JSON.parse(result.content[0].text); }
    catch (_) { throw new Error("No se pudo leer la respuesta de la herramienta."); }
  }

  async tool(name, args) {
    if (!this.ready) throw new Error("Conecta primero con la biblioteca de esta bóveda.");
    if (!REQUIRED_TOOLS.includes(name)) throw new Error("Esta herramienta no está habilitada en el complemento.");
    if (name === "grimore_search" && this.info.model_execution.embeddings.mode !== "local" && !this.settings.allowRemoteEmbeddings) {
      throw new Error("La búsqueda usaría embeddings remotos. Autorízalo expresamente en los ajustes o usa un servicio local.");
    }
    const connection = this.child;
    const result = await this.request("tools/call", { name, arguments: args });
    // Transport may already have resolved when disconnect clears its pending
    // map. Never publish that response into a closed or replacement session.
    if (!this.ready || this.child !== connection) throw new Error("La conexión se cerró o cambió. Repite la consulta.");
    return this.decodeTool(result);
  }

  close(message = "Conexión cerrada.", kill = true) {
    const child = this.child;
    this.child = null;
    this.ready = false;
    this.info = null;
    this.buffer = "";
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error(message));
    }
    this.pending.clear();
    if (kill && child && child.exitCode === null) {
      try { child.kill("SIGTERM"); } catch (_) { /* Already gone. */ }
      const timer = setTimeout(() => {
        if (child.exitCode === null) { try { child.kill("SIGKILL"); } catch (_) { /* Already gone. */ } }
      }, 2000);
      timer.unref?.();
      child.once("exit", () => clearTimeout(timer));
    }
    this.onState();
  }
}

function addText(container, tag, text, cls) {
  const element = container.createEl(tag, cls ? { cls } : {});
  element.textContent = String(text ?? ""); // Treat all model/index strings as inert text.
  return element;
}

function addButton(container, text, callback) {
  const button = addText(container, "button", text);
  button.type = "button";
  button.addEventListener("click", callback);
  return button;
}

function extractionText(extraction) {
  if (!extraction) return "Sin diagnóstico de extracción indexada.";
  if (extraction.status === "partial") return "Extracción parcial. Páginas sin texto: " + (extraction.missing_pages || []).join(", ");
  if (extraction.status === "unknown") return "Cobertura de extracción desconocida: índice antiguo.";
  return "Estado de extracción indexada: " + extraction.status + ". No certifica el respaldo de una afirmación.";
}

class SourceModal extends Modal {
  constructor(plugin, detail) { super(plugin.app); this.plugin = plugin; this.detail = detail; }
  onOpen() {
    this.plugin.modals?.add(this);
    this.contentEl.addClass("grimore-source-modal");
    const source = this.detail;
    addText(this.contentEl, "h2", source.document?.title || "Pasaje indexado");
    const states = { current: "Archivo actual coincide con el indexado", modified: "El archivo cambió después de indexarse", missing: "El archivo ya no está disponible", unavailable: "No se puede acceder al archivo", unverified: "Versión del archivo sin comprobar" };
    addText(this.contentEl, "p", states[source.source_state] || "Estado de fuente desconocido");
    addText(this.contentEl, "p", source.document?.path || "Sin ruta segura", "grimore-note");
    if (source.locator?.page) addText(this.contentEl, "p", "Página física: " + source.locator.page);
    if (source.locator?.heading) addText(this.contentEl, "p", "Sección indexada: " + source.locator.heading);
    addText(this.contentEl, "p", "Fragmento indexado: " + source.locator?.chunk_index, "grimore-note");
    if (source.document?.format === "pdf") addText(this.contentEl, "p", extractionText(source.indexed_extraction));
    addText(this.contentEl, "p", source.passage?.completeness === "full" ? "Pasaje almacenado completo; no es el documento entero." : "Solo se conserva un extracto, no el pasaje completo.", "grimore-note");
    addText(this.contentEl, "pre", source.passage?.text || "Sin texto almacenado", "grimore-passage");
    if (source.document?.format === "md" && source.document.path && !["missing", "unavailable"].includes(source.source_state)) {
      addButton(this.contentEl, "Abrir archivo actual", () => this.plugin.openDocument(source.document, source).catch(() => new Notice("No se pudo abrir la nota existente de esta bóveda.")));
    } else addText(this.contentEl, "p", "Apertura integrada disponible solo para notas Markdown existentes.", "grimore-note");
  }
  onClose() { this.contentEl.empty(); this.detail = null; this.plugin.modals?.delete(this); }
}

class GrimoreView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf); this.plugin = plugin; this.closed = false; this.busy = false; this.generation = 0; this.reviewGeneration = 0;
  }
  getViewType() { return VIEW; }
  getDisplayText() { return "Grimore"; }
  getIcon() { return "book-open"; }
  async onOpen() {
    this.closed = false;
    this.plugin.views.add(this);
    const el = this.contentEl;
    el.empty(); el.addClass("grimore-readonly-panel");
    addText(el, "h2", "Grimore");
    addText(el, "p", "Solo consulta. No edita notas ni confirma alertas.", "grimore-note");
    const controls = el.createDiv({ cls: "grimore-controls" });
    this.connectButton = addButton(controls, "Conectar", () => this.run(() => this.plugin.connect()));
    this.disconnectButton = addButton(controls, "Desconectar", () => this.plugin.client.close());
    this.connection = addText(el, "p", "");
    this.connection.setAttribute("role", "status");
    this.progress = addText(el, "p", "", "grimore-note");
    this.progress.setAttribute("role", "status");
    this.error = addText(el, "p", "", "grimore-error");
    this.error.setAttribute("role", "alert");
    addText(el, "h3", "Buscar pasajes");
    const form = el.createEl("form", { cls: "grimore-controls" });
    this.query = form.createEl("input", { type: "search", placeholder: "Buscar en esta bóveda" });
    this.query.setAttribute("aria-label", "Consulta de búsqueda");
    this.query.maxLength = 1000;
    this.query.required = true;
    this.searchButton = addText(form, "button", "Buscar");
    this.searchButton.type = "submit";
    this.searchResults = el.createDiv();
    form.addEventListener("submit", event => {
      event.preventDefault();
      const query = this.query.value.trim();
      if (query) this.run(async isCurrent => {
        const result = await this.plugin.client.tool("grimore_search", { query, top_k: 10 });
        if (isCurrent()) this.renderSearch(result, query);
      });
    });
    addText(el, "h3", "Alertas para revisión");
    const review = el.createDiv({ cls: "grimore-controls" });
    this.kind = review.createEl("select");
    this.kind.setAttribute("aria-label", "Tipo de alerta");
    for (const [value, label] of [["", "Todas"], ...Object.entries(KINDS)]) {
      const option = addText(this.kind, "option", label); option.value = value;
    }
    this.reviewButton = addButton(review, "Consultar alertas", () => this.loadReview(0));
    this.reviewResults = el.createDiv();
    this.kind.addEventListener("change", () => { this.reviewGeneration++; this.reviewResults.empty(); });
    this.refreshConnection();
  }
  refreshConnection() {
    if (this.closed || !this.connection) return;
    const client = this.plugin.client;
    this.progress.textContent = this.busy ? "Procesando la consulta… Puedes desconectar para detener el proceso local." : "";
    this.connection.textContent = client.ready
      ? "Conectado a esta bóveda. Embeddings de búsqueda: " + (client.info.model_execution.embeddings.mode === "local" ? "locales." : "remotos; requieren autorización en ajustes.")
      : client.child ? "Comprobando servidor y biblioteca…" : "Desconectado. Configura el ejecutable y pulsa Conectar.";
    this.connectButton.disabled = this.busy || Boolean(client.child);
    this.disconnectButton.disabled = !client.child;
    this.searchButton.disabled = this.busy || !client.ready;
    this.reviewButton.disabled = this.busy || !client.ready;
    if (!client.ready) {
      this.generation++;
      this.searchResults.empty(); this.reviewResults.empty();
    }
  }
  async run(job) {
    if (this.busy || this.closed) return;
    this.busy = true;
    this.error.textContent = "";
    this.refreshConnection();
    const generation = this.generation, connection = this.plugin.client.child;
    const isCurrent = () => !this.closed && this.generation === generation
      && this.plugin.client.ready && this.plugin.client.child === connection;
    try { await job(isCurrent); }
    catch (error) { if (!this.closed) this.error.textContent = error.message; }
    finally { this.busy = false; this.refreshConnection(); }
  }
  renderSearch(payload, query = "") {
    if (this.closed || !this.plugin.client.ready) return;
    if (!Array.isArray(payload?.hits)) throw new Error("La búsqueda no devolvió una lista válida.");
    this.searchResults.empty();
    if (query) addText(this.searchResults, "p", "Resultados para: " + query, "grimore-note");
    if (!payload.hits.length) addText(this.searchResults, "p", "No se encontraron pasajes para esta consulta.");
    for (const hit of payload.hits) {
      const card = this.searchResults.createDiv({ cls: "grimore-card" });
      addText(card, "h4", hit.title || "Documento");
      addText(card, "p", hit.snippet);
      if (hit.source_id) addButton(card, "Ver pasaje indexado", () => this.source(hit.source_id, hit.indexed_file_hash));
      else addText(card, "p", "Sin identidad de pasaje verificable. Reindexa desde Grimore si es necesario.", "grimore-note");
    }
  }
  source(id, fileHash) {
    return this.run(async isCurrent => {
      const args = { source_id: id };
      if (typeof fileHash === "string" && /^[0-9a-f]{64}$/.test(fileHash)) args.expected_indexed_file_hash = fileHash;
      const result = await this.plugin.client.tool("grimore_get_source", args);
      if (!isCurrent()) return;
      if (!result?.found) throw new Error(result?.reason === "source_version_changed" ? "El índice cambió. Repite la búsqueda o actualiza las alertas." : "Ese pasaje ya no está en el índice. Repite la consulta.");
      if (!result.source?.document || !result.source?.passage) throw new Error("El pasaje recibido está incompleto.");
      if (!this.closed && this.plugin.client.ready) new SourceModal(this.plugin, result.source).open();
    });
  }
  loadReview(offset) {
    const kind = this.kind.value;
    const reviewGeneration = this.reviewGeneration;
    return this.run(async isCurrent => {
      const args = { limit: 20, offset };
      if (kind) args.kind = kind;
      const result = await this.plugin.client.tool("grimore_review_inbox", args);
      if (!isCurrent()) return;
      if (!Array.isArray(result?.items) || !result.delivery) throw new Error("Las alertas recibidas están incompletas.");
      if (this.closed || !this.plugin.client.ready) return;
      if (this.kind.value !== kind || this.reviewGeneration !== reviewGeneration) throw new Error("El filtro cambió; vuelve a consultar las alertas.");
      this.reviewResults.empty();
      addText(this.reviewResults, "p", `${result.delivery.returned} de ${result.delivery.matched} candidatos de este conjunto. Duplicados cercanos: hasta 30 pares. La paginación no es una instantánea fija.`, "grimore-note");
      if (!result.items.length) addText(this.reviewResults, "p", "No hay candidatos activos en esta página; no es una auditoría exhaustiva.");
      for (const item of result.items) {
        const card = this.reviewResults.createDiv({ cls: "grimore-card" });
        addText(card, "h4", KINDS[item.kind] || "Candidato");
        addText(card, "p", item.reason);
        addText(card, "p", "Señal para revisión humana, no un veredicto.", "grimore-note");
        const coverage = card.createEl("details");
        addText(coverage, "summary", "Cobertura y límites de la señal");
        addText(coverage, "pre", JSON.stringify({ coverage: item.coverage, delivery: item.delivery }, null, 2), "grimore-passage");
        for (const doc of item.documents || []) {
          addText(card, "p", doc.title || doc.path || "Documento");
          if (doc.path?.toLowerCase().endsWith(".md")) {
            addButton(card, "Abrir nota actual", () => this.plugin.openDocument(doc).catch(() => new Notice("No se pudo abrir la nota existente de esta bóveda.")));
          }
        }
        for (const passage of item.passages || []) {
          addText(card, "pre", passage.text, "grimore-passage");
          if (passage.text_truncated || !passage.complete) addText(card, "p", "Texto parcial o recortado para esta vista.", "grimore-note");
          if (passage.source_id) addButton(card, "Ver pasaje indexado", () => this.source(passage.source_id, passage.indexed_file_hash));
          else addText(card, "p", "Afirmación extraída; no identifica un fragmento indexado independiente.", "grimore-note");
        }
      }
      if (offset > 0) addButton(this.reviewResults, "Primera página", () => this.loadReview(0));
      if (result.delivery.has_more && Number.isInteger(result.delivery.next_offset)) {
        addButton(this.reviewResults, "Siguiente página", () => this.loadReview(result.delivery.next_offset));
      }
    });
  }
  async onClose() {
    this.closed = true; this.generation++;
    this.plugin.views.delete(this);
    this.contentEl.empty();
    if (!this.plugin.views.size) this.plugin.client.close();
  }
}

class GrimoreSettings extends PluginSettingTab {
  display() {
    const el = this.containerEl;
    el.empty();
    addText(el, "h2", "Grimore: consultas locales");
    addText(el, "p", "No se inicia ningún proceso hasta pulsar Conectar. Cambiar ajustes desconecta. El ejecutable elegido debe ser de confianza.");
    for (const [key, name, description] of [
      ["executable", "Ejecutable de Grimore", "Ruta absoluta al ejecutable instalado, sin argumentos ni órdenes de shell."],
      ["cwd", "Directorio de configuración", "Ruta absoluta a la carpeta con grimore.toml. La biblioteca debe coincidir con esta bóveda."],
      ["profile", "Perfil opcional", "Nombre explícito del perfil de grimore.toml. Vacío usa la configuración principal."],
    ]) {
      new Setting(el).setName(name).setDesc(description).addText(text => text.setValue(this.plugin.settings[key]).onChange(async value => {
        this.plugin.settings[key] = value.trim(); await this.plugin.saveSettings();
      }));
    }
    new Setting(el).setName("Tiempo de espera (segundos)").setDesc("Entre 5 y 300. Un timeout cierra el proceso local; no garantiza cancelar un modelo externo.").addText(text => text.setValue(String(this.plugin.settings.timeoutSeconds)).onChange(async value => {
      this.plugin.settings.timeoutSeconds = Number(value); await this.plugin.saveSettings();
    }));
    new Setting(el).setName("Autorizar embeddings remotos").setDesc("Desactivado por defecto. Si Grimore usa embeddings remotos, la búsqueda envía tu consulta a ese servicio. Los pasajes y alertas se leen localmente.").addToggle(toggle => toggle.setValue(this.plugin.settings.allowRemoteEmbeddings).onChange(async value => {
      this.plugin.settings.allowRemoteEmbeddings = value; await this.plugin.saveSettings();
    }));
  }
}

class GrimorePlugin extends Plugin {
  async onload() {
    const saved = await this.loadData();
    this.settings = { ...DEFAULTS };
    if (saved && typeof saved === "object") {
      for (const key of Object.keys(DEFAULTS)) if (typeof saved[key] === typeof DEFAULTS[key]) this.settings[key] = saved[key];
    }
    this.views = new Set();
    this.modals = new Set();
    this.client = new MCPClient({ onState: () => {
      if (!this.client.ready) for (const modal of [...this.modals]) modal.close();
      for (const view of this.views) view.refreshConnection();
    } });
    this.registerView(VIEW, leaf => new GrimoreView(leaf, this));
    this.addCommand({ id: "open-panel", name: "Abrir panel de consultas", callback: () => this.activate().catch(() => new Notice("No se pudo abrir el panel de Grimore.")) });
    this.addCommand({ id: "disconnect", name: "Desconectar servidor local", callback: () => this.client.close() });
    this.addSettingTab(new GrimoreSettings(this.app, this));
  }
  async saveSettings() { this.client.close(); await this.saveData(this.settings); }
  async connect() {
    const adapter = this.app.vault.adapter;
    if (typeof adapter.getBasePath !== "function") throw new Error("Este complemento necesita Obsidian de escritorio y una bóveda local.");
    return this.client.connect({ ...this.settings }, adapter.getBasePath());
  }
  async activate() {
    let leaf = this.app.workspace.getLeavesOfType(VIEW)[0];
    if (!leaf) {
      leaf = this.app.workspace.getRightLeaf(false);
      if (!leaf) throw new Error("No hay panel disponible.");
      await leaf.setViewState({ type: VIEW, active: true });
    }
    await this.app.workspace.revealLeaf(leaf);
  }
  async openDocument(document, source = null) {
    if (!this.client.ready) throw new Error("La biblioteca no está conectada.");
    const connection = this.client.child;
    const relative = safeRelativePath(document.path);
    const root = fs.realpathSync.native(this.app.vault.adapter.getBasePath());
    if (fingerprint(root) !== this.client.info.vault_identity.fingerprint) throw new Error("La bóveda cambió.");
    const absolute = containedFile(root, relative);
    const file = this.app.vault.getAbstractFileByPath(relative);
    if (!(file instanceof TFile) || file.extension.toLowerCase() !== "md") throw new Error("La nota Markdown ya no existe.");
    let heading = null;
    if (source?.source_state === "current" && /^[0-9a-f]{64}$/.test(source.indexed_file_hash || "")
        && await hashFile(absolute) === source.indexed_file_hash) {
      const headings = this.app.metadataCache.getFileCache(file)?.headings || [];
      const matches = headings.filter(entry => entry.heading === source.locator?.heading);
      if (matches.length === 1) heading = matches[0];
    }
    if (!this.client.ready || this.client.child !== connection) throw new Error("La conexión se cerró o cambió.");
    const leaf = this.app.workspace.getLeaf("tab");
    await leaf.openFile(file); // Existing exact TFile: no title fallback and no note creation.
    if (heading && leaf.view instanceof MarkdownView) {
      const position = heading.position.start;
      leaf.view.editor.setCursor({ line: position.line, ch: position.col });
      leaf.view.editor.scrollIntoView({ from: { line: position.line, ch: position.col }, to: { line: position.line, ch: position.col } }, true);
    } else if (source?.locator?.heading) new Notice("Se abrió el documento actual; no se seleccionó un encabezado ambiguo o desactualizado.");
  }
  onunload() {
    this.client.close();
    for (const modal of [...this.modals]) modal.close();
    for (const leaf of this.app.workspace.getLeavesOfType(VIEW)) leaf.detach();
  }
}

module.exports = GrimorePlugin;
// Pure helpers exposed for local automated checks; no extra runtime entry point.
module.exports._internals = { MCPClient, fingerprint, validateSettings, safeRelativePath, containedFile, GrimoreView, SourceModal, addText, DEFAULTS, MAX_FRAME };
