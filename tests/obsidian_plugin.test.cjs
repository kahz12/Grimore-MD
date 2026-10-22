"use strict";
const assert = require("node:assert/strict");
const { test } = require("node:test");
const { EventEmitter } = require("node:events");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const crypto = require("node:crypto");
const os = require("node:os");
const { spawn: actualSpawn } = require("node:child_process");

const filename = path.join(__dirname, "../integrations/obsidian/grimore-readonly/main.js");
const script = fs.readFileSync(filename, "utf8");
const manifest = JSON.parse(fs.readFileSync(path.join(path.dirname(filename), "manifest.json"), "utf8"));

class Element {
  constructor(tag = "div") { this.tag = tag; this.children = []; this.events = {}; this.textContent = ""; this.value = ""; this.attrs = {}; }
  createEl(tag, opts = {}) { const el = new Element(tag); Object.assign(el, opts); this.children.push(el); return el; }
  createDiv(opts) { return this.createEl("div", opts); }
  empty() { this.children = []; }
  addClass(cls) { this.cls = cls; }
  setAttribute(key, value) { this.attrs[key] = value; }
  addEventListener(event, callback) { this.events[event] = callback; }
  flatten() { return [this, ...this.children.flatMap(el => el.flatten())]; }
}

function load() {
  const notices = [];
  class Plugin {
    constructor(app) { this.app = app; this.registered = []; }
    async loadData() { return this.saved; }
    async saveData(data) { this.saved = data; }
    registerView(...args) { this.registered.push(args); }
    addCommand(command) { this.registered.push(command); }
    addSettingTab(tab) { this.tab = tab; }
  }
  class Modal {
    constructor(app) { this.app = app; this.contentEl = new Element(); }
    open() { this.onOpen(); this.app.modal = this; }
    close() { this.onClose(); }
  }
  class ItemView { constructor(leaf) { this.leaf = leaf; this.contentEl = new Element(); } }
  class TFile { constructor(relative) { this.path = relative; this.extension = path.extname(relative).slice(1); } }
  class MarkdownView {}
  const obsidian = { Plugin, ItemView, Modal, TFile, MarkdownView,
    Notice: class { constructor(text) { notices.push(text); } },
    PluginSettingTab: class {}, Setting: class {},
  };
  const module = { exports: {} };
  vm.runInNewContext(script, {
    module, exports: module.exports, require: name => name === "obsidian" ? obsidian : require(name),
    Buffer, process, setTimeout, clearTimeout,
  }, { filename });
  return { Plugin: module.exports, ...module.exports._internals, obsidian, notices };
}

const loaded = load();
const { MCPClient, fingerprint, DEFAULTS, safeRelativePath, containedFile, validateSettings } = loaded;
const settings = { ...DEFAULTS, executable: path.resolve("grimore"), cwd: path.resolve("config") };
const root = path.resolve("vault");

function toolResult(payload) { return { content: [{ type: "text", text: JSON.stringify(payload) }], isError: false }; }

function childProcess(handler) {
  const child = new EventEmitter();
  child.stdout = new EventEmitter(); child.stdout.setEncoding = () => {};
  child.stderr = new EventEmitter(); child.stdin = new EventEmitter();
  child.exitCode = null; child.kills = []; child.sent = [];
  child.stdin.write = frame => {
    const message = JSON.parse(frame);
    child.sent.push(message);
    if (message.id !== undefined) queueMicrotask(() => {
      const result = handler(message, child);
      if (result !== undefined) child.stdout.emit("data", JSON.stringify({ jsonrpc: "2.0", id: message.id, result }) + "\n");
    });
  };
  child.kill = signal => { child.kills.push(signal); queueMicrotask(() => { child.exitCode = 0; child.emit("exit", 0); }); };
  return child;
}

function clientFixture({ vaultRoot = root, mode = "local", content, override, onState } = {}) {
  let launch;
  const info = { contract_version: 1, vault_identity: { algorithm: "sha256-local-path-v1", fingerprint: fingerprint(vaultRoot) }, model_execution: { embeddings: { mode } } };
  const child = childProcess((request, child) => {
    if (override) { const value = override(request, child); if (value !== null) return value; }
    if (request.method === "initialize") return { serverInfo: { name: "grimore" }, protocolVersion: "2024-11-05" };
    if (request.method === "tools/list") return { tools: ["grimore_library_info", "grimore_search", "grimore_get_source", "grimore_review_inbox"].map(name => ({ name })) };
    if (request.params.name === "grimore_library_info") return toolResult(info);
    return toolResult(content || { hits: [] });
  });
  const client = new MCPClient({ spawnProcess: (...args) => { launch = args; return child; }, realpath: value => value, onState });
  return { client, child, info, launch: () => launch };
}

test("manifest is desktop-only and ships standalone install files", () => {
  assert.equal(manifest.id, "grimore-readonly");
  assert.equal(manifest.isDesktopOnly, true);
  assert.ok(fs.existsSync(path.join(path.dirname(filename), "styles.css")));
  assert.doesNotThrow(() => new vm.Script(script));
});

test("construction and loading do not spawn a server or query anything", async () => {
  const f = clientFixture();
  assert.equal(f.launch(), undefined);
  const p = new loaded.Plugin({});
  await p.onload();
  assert.equal(p.client.child, null);
  assert.equal(p.client.ready, false);
});

test("connection performs metadata-only handshake, no shell or inherited profile", async t => {
  const f = clientFixture(); t.after(() => f.client.close());
  await f.client.connect(settings, root);
  assert.equal(f.client.ready, true);
  const [exe, args, options] = f.launch();
  assert.equal(exe, settings.executable);
  assert.deepEqual(Array.from(args), ["mcp"]);
  assert.equal(options.shell, false);
  assert.equal(options.cwd, settings.cwd);
  assert.equal(options.env.GRIMORE_PROFILE, undefined);
  assert.deepEqual(f.child.sent.map(r => r.method), ["initialize", "notifications/initialized", "tools/list", "tools/call"]);
  assert.equal(f.child.sent.at(-1).params.name, "grimore_library_info");
  assert.equal(f.child.sent[0].params.clientInfo.version, manifest.version);
});

test("explicit profile is one argument, never a shell command", async t => {
  const f = clientFixture(); t.after(() => f.client.close());
  await f.client.connect({ ...settings, profile: "personal-notes" }, root);
  assert.deepEqual(Array.from(f.launch()[1]), ["--profile", "personal-notes", "mcp"]);
});

test("concurrent connect calls share one startup", async t => {
  const f = clientFixture(); t.after(() => f.client.close());
  const one = f.client.connect(settings, root), two = f.client.connect(settings, root);
  assert.equal(one, two);
  await one;
  assert.equal(f.child.sent.filter(r => r.method === "initialize").length, 1);
});

test("wrong vault closes process before any content request", async () => {
  const f = clientFixture({ vaultRoot: root + "-other" });
  await assert.rejects(f.client.connect(settings, root), /no coincide/);
  assert.equal(f.client.ready, false);
  assert.deepEqual(f.child.sent.filter(r => r.method === "tools/call").map(r => r.params.name), ["grimore_library_info"]);
  assert.ok(f.child.kills.includes("SIGTERM"));
});

for (const [name, override] of Object.entries({
  wrongServer: request => request.method === "initialize" ? { serverInfo: { name: "other" }, protocolVersion: "2024-11-05" } : null,
  oldProtocol: request => request.method === "initialize" ? { serverInfo: { name: "grimore" }, protocolVersion: "old" } : null,
  oldTools: request => request.method === "tools/list" ? { tools: [] } : null,
  missingMode: request => request.params?.name === "grimore_library_info" ? toolResult({ contract_version: 1, vault_identity: { algorithm: "sha256-local-path-v1", fingerprint: fingerprint(root) } }) : null,
})) {
  test(`incompatible handshake ${name} is rejected`, async () => {
    const f = clientFixture({ override });
    await assert.rejects(f.client.connect(settings, root));
    assert.equal(f.client.child, null);
  });
}

test("remote embeddings require explicit consent; sources/review remain usable", async t => {
  const f = clientFixture({ mode: "remote" }); t.after(() => f.client.close());
  await f.client.connect(settings, root);
  const count = f.child.sent.length;
  await assert.rejects(f.client.tool("grimore_search", { query: "Private question" }), /remotos/);
  assert.equal(f.child.sent.length, count);
  await f.client.tool("grimore_get_source", { source_id: "id" });
  await f.client.tool("grimore_review_inbox", {});
});

test("authorized remote embedding query is sent only on demand", async t => {
  const f = clientFixture({ mode: "remote" }); t.after(() => f.client.close());
  await f.client.connect({ ...settings, allowRemoteEmbeddings: true }, root);
  await f.client.tool("grimore_search", { query: "on demand", top_k: 10 });
  assert.equal(f.child.sent.at(-1).params.arguments.query, "on demand");
});

for (const tool of ["grimore_ask", "grimore_get_note", "grimore_scan", "confirm", "grimore_connect"]) {
  test(`client does not expose ${tool}`, async t => {
    const f = clientFixture(); t.after(() => f.client.close());
    await f.client.connect(settings, root);
    const count = f.child.sent.length;
    await assert.rejects(f.client.tool(tool, {}), /no está habilitada/);
    assert.equal(f.child.sent.length, count);
  });
}

test("NDJSON handles Unicode fragmented across frames and multiple response lines", async t => {
  const f = clientFixture(); t.after(() => f.client.close());
  await f.client.connect(settings, root);
  f.child.stdin.write = frame => { f.child.sent.push(JSON.parse(frame)); };
  const promise = f.client.tool("grimore_search", { query: "á" });
  const id = f.child.sent.at(-1).id;
  const response = JSON.stringify({ jsonrpc: "2.0", id, result: toolResult({ hits: [{ title: "á & <script>" }] }) }) + "\n";
  f.child.stdout.emit("data", response.slice(0, 20));
  f.child.stdout.emit("data", response.slice(20) + JSON.stringify({ jsonrpc: "2.0", method: "notifications/progress" }) + "\n");
  assert.equal((await promise).hits[0].title, "á & <script>");
  assert.equal(f.client.pending.size, 0);
});

test("out-of-vault symlinks are rejected at open time", () => {
  assert.throws(() => containedFile(root, "valid.md", () => path.resolve("other/private.md")), /fuera/);
});

for (const value of ["", "../bad.md", "/outside.md", "a/../b.md", "a//b.md", "a\\b.md", "file#Heading.md", "https://evil.md", "C:secret.md", "note\0.md"]) {
  test(`reject unsafe note path ${JSON.stringify(value)}`, () => assert.throws(() => safeRelativePath(value)));
}

test("fingerprint matches documented Python byte protocol including Windows", () => {
  const expected = crypto.createHash("sha256").update("grimore-vault-v1\0" + root).digest("hex");
  assert.equal(fingerprint(root), expected);
  assert.equal(fingerprint("C:/Vault/Notes", "win32"), crypto.createHash("sha256").update("grimore-vault-v1\0c:\\vault\\notes").digest("hex"));
});

for (const changes of [
  { executable: "grimore; evil" }, { cwd: "relative" }, { executable: settings.executable + "\n" },
  { profile: "--unsafe" }, { profile: "foo; evil" }, { profile: "x".repeat(81) },
  { timeoutSeconds: 0 }, { timeoutSeconds: 301 }, { timeoutSeconds: 5.5 }, { allowRemoteEmbeddings: "false" },
]) {
  test(`invalid settings rejected before process launch ${JSON.stringify(changes)}`, async () => {
    const f = clientFixture();
    await assert.rejects(f.client.connect({ ...settings, ...changes }, root));
    assert.equal(f.launch(), undefined);
  });
}

test("disconnect rejects pending requests and prevents stale process results", async () => {
  const f = clientFixture();
  await f.client.connect(settings, root);
  f.child.stdin.write = () => {};
  const waiting = f.client.tool("grimore_search", { query: "q" });
  f.client.close();
  await assert.rejects(waiting, /cerrada/);
  f.child.stdout.emit("data", "not JSON\n");
  assert.equal(f.client.ready, false);
  assert.equal(f.client.pending.size, 0);
});

for (const reconnect of [false, true]) {
  test(`a received tool response cannot survive a session change (reconnect=${reconnect})`, async t => {
    const f = clientFixture(); t.after(() => f.client.close());
    await f.client.connect(settings, root);
    f.child.stdin.write = frame => { f.child.sent.push(JSON.parse(frame)); };
    const waiting = f.client.tool("grimore_search", { query: "old session" });
    const id = f.child.sent.at(-1).id;
    // Resolve transport synchronously, then disconnect before the awaiting
    // tool continuation can decode/publish its private content.
    f.child.stdout.emit("data", JSON.stringify({ jsonrpc: "2.0", id,
      result: toolResult({ hits: [{ title: "private-old-session" }] }) }) + "\n");
    f.client.close();
    if (reconnect) { f.client.child = childProcess(() => undefined); f.client.ready = true; f.client.info = f.info; }
    await assert.rejects(waiting, /conexión|Conexión/);
  });
}

function deferredView() {
  let resolve;
  const client = { ready: true, child: {},
    info: { model_execution: { embeddings: { mode: "local" } } },
    tool: () => new Promise(done => { resolve = done; }), close() {},
  };
  const plugin = { app: {}, views: new Set(), modals: new Set(), client };
  const view = new loaded.GrimoreView({}, plugin);
  return { view, plugin, client, resolve: result => resolve(result) };
}

for (const operation of ["search", "source", "review"]) {
  for (const transition of ["reconnect", "reopen"]) {
  test(`a late ${operation} result is not rendered after ${transition}`, async () => {
    const f = deferredView(); await f.view.onOpen();
    let waiting;
    if (operation === "search") {
      f.view.query.value = "old query";
      f.view.contentEl.flatten().find(el => el.tag === "form").events.submit({ preventDefault() {} });
    } else waiting = operation === "source" ? f.view.source("old-id") : f.view.loadReview(0);
    if (transition === "reconnect") {
      f.client.ready = false; f.view.refreshConnection();
      f.client.child = {}; f.client.ready = true; f.view.refreshConnection();
    } else {
      await f.view.onClose(); await f.view.onOpen();
    }
    const result = operation === "search" ? { hits: [{ title: "private-old-session" }] }
      : operation === "source" ? { found: true, source: { document: { title: "private-old-session", format: "pdf" }, passage: { text: "old passage" } } }
      : { items: [{ reason: "private-old-session" }], delivery: { matched: 1, returned: 1 } };
    f.resolve(result);
    if (waiting) await waiting;
    else await new Promise(done => setImmediate(done));
    assert.equal(f.view.searchResults.children.length, 0);
    assert.equal(f.view.reviewResults.children.length, 0);
    assert.equal(f.plugin.modals.size, 0);
  });
  }
}

test("returning to the same alert filter does not resurrect its old response", async () => {
  const f = deferredView(); await f.view.onOpen();
  f.view.kind.value = "contradiction";
  const waiting = f.view.loadReview(0);
  f.view.kind.value = "duplicate_exact"; f.view.kind.events.change();
  f.view.kind.value = "contradiction"; f.view.kind.events.change();
  f.resolve({ items: [{ reason: "private-old-filter" }], delivery: { matched: 1, returned: 1 } });
  await waiting;
  assert.equal(f.view.reviewResults.children.length, 0);
  assert.match(f.view.error.textContent, /filtro cambió/);
});

test("transport timeout tears down client and does not promise model cancellation", async () => {
  const f = clientFixture();
  await f.client.connect(settings, root);
  f.child.stdin.write = () => {};
  await assert.rejects(f.client.request("ping", {}, 5), /puede seguir trabajando/);
  assert.equal(f.client.child, null);
});

test("only one request can be in flight; no unbounded request queue", async () => {
  const f = clientFixture();
  await f.client.connect(settings, root);
  f.child.stdin.write = () => {};
  const first = f.client.tool("grimore_search", { query: "one" });
  await assert.rejects(f.client.tool("grimore_review_inbox", {}), /en curso/);
  f.client.close(); await assert.rejects(first);
});

for (const frame of ["not JSON\n", "null\n", '{"jsonrpc":"bad","id":9}\n', '{"jsonrpc":"2.0","id":"1","result":{}}\n', "x".repeat(8 * 1024 * 1024 + 1)]) {
  test(`malformed/oversized transport rejected (${frame.length} bytes)`, async () => {
    const f = clientFixture();
    await f.client.connect(settings, root);
    f.child.stdin.write = () => {};
    const pending = f.client.tool("grimore_review_inbox", {});
    f.child.stdout.emit("data", frame);
    await assert.rejects(pending);
    assert.equal(f.client.child, null);
  });
}

test("stdout close and spawn error fail cleanly", async () => {
  const f = clientFixture();
  await f.client.connect(settings, root);
  f.child.stdin.write = () => {};
  const pending = f.client.tool("grimore_review_inbox", {});
  f.child.stdout.emit("end");
  await assert.rejects(pending, /cerró/);
  const failing = new MCPClient({ realpath: x => x, spawnProcess: () => { throw new Error("secret process path"); } });
  await assert.rejects(failing.connect(settings, root), /No se pudo iniciar/);
});

test("server error details are not rendered or retained", async t => {
  const f = clientFixture(); t.after(() => f.client.close());
  await f.client.connect(settings, root);
  f.child.stdin.write = frame => {
    const request = JSON.parse(frame);
    queueMicrotask(() => f.child.stdout.emit("data", JSON.stringify({ jsonrpc: "2.0", id: request.id, error: { message: "API_TOKEN=private" } }) + "\n"));
  };
  await assert.rejects(f.client.tool("grimore_review_inbox", {}), error => !error.message.includes("private"));
  assert.equal(f.client.buffer, "");
});

test("tool content errors are sanitized and malformed content fails", () => {
  const client = new MCPClient();
  assert.throws(() => client.decodeTool({ isError: true, content: [{ text: "secret" }] }), /falló/);
  for (const response of [null, {}, { content: [{ type: "image" }] }, { content: [{ type: "text", text: "not JSON" }] }]) {
    assert.throws(() => client.decodeTool(response));
  }
});

test("search/results/source renderer never interprets untrusted HTML", async () => {
  const app = {};
  const plugin = { app, views: new Set(), client: { ready: true, info: { model_execution: { embeddings: { mode: "local" } } }, close() {} }, connect() {} };
  const view = new loaded.GrimoreView({}, plugin);
  await view.onOpen();
  view.renderSearch({ hits: [{ title: "<script>evil()</script>", snippet: "<img onerror=evil()>", source_id: "exact" }] });
  assert.ok(view.searchResults.flatten().some(el => el.textContent === "<script>evil()</script>"));
  assert.ok(view.searchResults.flatten().some(el => el.textContent === "<img onerror=evil()>"));
  assert.ok(!view.searchResults.flatten().some(el => el.tag === "script" || el.tag === "img"));
  const modal = new loaded.SourceModal(plugin, { document: { title: "<iframe>", format: "pdf", path: "a.pdf" }, passage: { text: "<script>", completeness: "full" }, source_state: "modified", locator: { page: 3 } });
  modal.open();
  assert.ok(modal.contentEl.flatten().some(el => el.textContent === "<script>"));
  assert.ok(!modal.contentEl.flatten().some(el => el.tag === "script"));
  modal.onClose();
  assert.equal(modal.detail, null);
});

test("connect errors remain visible when lifecycle refreshes the view", async () => {
  const plugin = { views: new Set(), client: { ready: false, child: null, close() {} }, connect() {} };
  const view = new loaded.GrimoreView({}, plugin);
  await view.onOpen();
  await view.run(async () => { view.refreshConnection(); throw new Error("Biblioteca incorrecta"); });
  assert.equal(view.error.textContent, "Biblioteca incorrecta");
  assert.equal(view.busy, false);
});

test("closing the last panel stops the child; unload stops it too", async () => {
  let closed = 0;
  const plugin = { views: new Set(), client: { ready: false, child: null, close() { closed++; } } };
  const view = new loaded.GrimoreView({}, plugin);
  await view.onOpen(); await view.onClose();
  assert.equal(closed, 1);
  assert.equal(view.contentEl.children.length, 0);
  const p = new loaded.Plugin({ workspace: { getLeavesOfType: () => [] } });
  await p.onload();
  p.client.close = () => { closed++; };
  p.onunload(); assert.equal(closed, 2);
});

function noteFixture(t, { headings = [1], sourceState = "current", differentHash = false } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "grimore-obsidian-test-"));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  fs.writeFileSync(path.join(dir, "note.md"), "# Heading\n\nText");
  const canonical = fs.realpathSync.native(dir);
  const file = new loaded.obsidian.TFile("note.md");
  const calls = [];
  const leaf = { async openFile(f) { calls.push(["open", f.path]); }, view: new loaded.obsidian.MarkdownView() };
  leaf.view.editor = { setCursor(p) { calls.push(["cursor", p.line]); }, scrollIntoView() {} };
  const p = new loaded.Plugin({
    vault: { adapter: { getBasePath: () => canonical }, getAbstractFileByPath: name => name === "note.md" ? file : null },
    metadataCache: { getFileCache: () => ({ headings: headings.map(line => ({ heading: "Heading", position: { start: { line, col: 0 } } })) }) },
    workspace: { getLeaf: () => leaf },
  });
  p.client = { ready: true, child: {}, info: { vault_identity: { fingerprint: fingerprint(canonical) } } };
  const source = { source_state: sourceState, locator: { heading: "Heading" }, indexed_file_hash: differentHash ? "a".repeat(64) : crypto.createHash("sha256").update("# Heading\n\nText").digest("hex") };
  return { plugin: p, source, calls, root: canonical };
}

test("opens the exact existing TFile, with a unique verified heading", async t => {
  const f = noteFixture(t);
  await f.plugin.openDocument({ path: "note.md" }, f.source);
  assert.deepEqual(f.calls, [["open", "note.md"], ["cursor", 1]]);
});

for (const options of [{ headings: [1, 5] }, { differentHash: true }, { sourceState: "modified" }]) {
  test(`does not guess repeated or outdated headings ${JSON.stringify(options)}`, async t => {
    const f = noteFixture(t, options);
    await f.plugin.openDocument({ path: "note.md" }, f.source);
    assert.deepEqual(f.calls, [["open", "note.md"]]);
  });
}

test("no note creation for missing paths; wrong vault cannot open a file", async t => {
  const f = noteFixture(t);
  await assert.rejects(f.plugin.openDocument({ path: "missing.md" }));
  f.plugin.client.info.vault_identity.fingerprint = "wrong";
  await assert.rejects(f.plugin.openDocument({ path: "note.md" }), /cambió/);
  assert.equal(f.calls.length, 0);
});

test("disconnect clears open source modals", async () => {
  const p = new loaded.Plugin({});
  await p.onload();
  const modal = new loaded.SourceModal(p, { document: { format: "pdf" }, passage: { text: "private" } });
  modal.open();
  assert.equal(p.modals.size, 1);
  p.client.close();
  assert.equal(p.modals.size, 0);
  assert.equal(modal.detail, null);
});

test("editor client interoperates with the actual Python MCP server", {
  skip: !process.env.GRIMORE_OBSIDIAN_TEST_PYTHON,
}, async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "grimore-mcp-bridge-"));
  const vault = path.join(dir, "vault"); fs.mkdirSync(vault);
  for (const name of ["note.md", "copy.md"]) fs.writeFileSync(path.join(vault, name), "# Heading\n\nTemporary evidence");
  const client = new MCPClient({ spawnProcess: (_exe, _args, options) => actualSpawn(
    process.env.GRIMORE_OBSIDIAN_TEST_PYTHON, ["-m", "tests.obsidian_mcp_fixture", vault],
    { ...options, cwd: path.resolve(__dirname, "..") },
  ) });
  t.after(async () => {
    const child = client.child;
    const exited = child && child.exitCode === null ? new Promise(resolve => child.once("exit", resolve)) : Promise.resolve();
    client.close(); await exited;
    fs.rmSync(dir, { recursive: true, force: true });
  });
  await client.connect({ ...settings }, vault);
  const result = await client.tool("grimore_search", { query: "Evidence", top_k: 1 });
  assert.equal(result.hits.length, 1);
  const hit = result.hits[0];
  assert.match(hit.source_id, /:chunk:1:/);
  const evidence = await client.tool("grimore_get_source", { source_id: hit.source_id, expected_indexed_file_hash: hit.indexed_file_hash });
  assert.equal(evidence.source.passage.text, "Exact indexed passage 1");
  assert.equal(evidence.source.source_state, "current");
  const inbox = await client.tool("grimore_review_inbox", { limit: 20 });
  assert.equal(inbox.items[0].kind, "duplicate_exact");
  assert.equal(inbox.items[0].documents[0].path.includes(vault), false);
  assert.equal(fs.readFileSync(path.join(vault, "note.md"), "utf8"), "# Heading\n\nTemporary evidence");
});
