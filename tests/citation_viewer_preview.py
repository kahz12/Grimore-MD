"""Read-only visual fixture for citation navigation; no Session, index or LLM.

Run manually with venv/bin/python tests/citation_viewer_preview.py. This is a
synthetic UI check, not a benchmark or a representative model answer.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8769
FIXTURE = r"""
window.fetch = async (url, options = {}) => {
    if (options.method && options.method !== 'GET') return new Response('{}', {status: 405});
    let data = {};
    if (String(url).endsWith('/api/health')) data = {ok: true, version: 'UI fixture', cognition: {mode: 'local'}};
    if (String(url).endsWith('/api/categories')) data = {categories: []};
    if (String(url).endsWith('/api/filter-options')) data = {categories: [], formats: [], tags: []};
    if (String(url).endsWith('/api/review-inbox')) data = {items: [], counts: {}};
    return new Response(JSON.stringify(data), {status: 200, headers: {'Content-Type': 'application/json'}});
};
document.addEventListener('DOMContentLoaded', () => {
    state.language = 'es';
    document.documentElement.lang = 'es';
    document.querySelector('#language-select').value = 'es';
    const label = 'Humedal & entorno#p.3';
    state.sourceDetails = [
        {id: 'fixture:0', label, citation_ref: 'cite:1', source_state: 'modified',
         document: {title: 'Humedal & entorno', path: 'informe.pdf', format: 'pdf'},
         locator: {page: 3, chunk_index: 0},
         passage: {text: 'Distractor: la meta global no describe el área aprobada.', completeness: 'full'}},
        {id: 'fixture:1', label, citation_ref: 'cite:2', source_state: 'missing',
         document: {title: 'Humedal & entorno', path: 'informe.pdf', format: 'pdf'},
         locator: {page: 3, chunk_index: 1},
         passage: {text: 'El área de intervención aprobada es de 37,5 hectáreas.', completeness: 'full'}},
    ];
    state.citationSelections = [
        {citation_index: 0, label, form: 'short', citation_ref: 'cite:2', source_detail_indices: [1], selection: 'exact'},
        {citation_index: 1, label, form: 'short', citation_ref: 'cite:1', source_detail_indices: [0], selection: 'exact'},
        {citation_index: 2, label, form: 'wikilink', citation_ref: null, source_detail_indices: [0,1], selection: 'ambiguous'},
    ];
    state.sourceLabels = [label, label];
    state.conversation = [{q: 'Prueba de interfaz', a: 'Respuesta sintética.', sources: [label]}];
    renderConversation();
    document.querySelector('#answer').hidden = false;
    document.querySelector('#answer-body').innerHTML = renderProse(
        'Prueba de interfaz, no respuesta del modelo.\n\n' +
        'Primera cita: debe abrir el segundo fragmento (37,5 hectáreas) [[' + label + ']].\n\n' +
        'Segunda cita: debe abrir el distractor [[' + label + ']].\n\n' +
        'Tercera cita: es ambigua y debe ofrecer ambos pasajes [[' + label + ']].');
    wireCitations();
    renderSources(state.sourceLabels, state.sourceDetails);
});
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            html = (ROOT / "grimore/api/templates/index.html").read_text()
            html = html.replace('    <script src="/static/app.js" defer></script>',
                                '    <script src="/static/app.js" defer></script>\n'
                                '    <script src="/fixture.js" defer></script>')
            data, content_type = html.encode(), "text/html; charset=utf-8"
        elif self.path == "/fixture.js":
            data, content_type = FIXTURE.encode(), "text/javascript; charset=utf-8"
        elif self.path.startswith("/static/"):
            target = (ROOT / "grimore/api" / self.path.lstrip("/")).resolve()
            static = (ROOT / "grimore/api/static").resolve()
            if not target.is_relative_to(static) or not target.is_file():
                self.send_error(404)
                return
            data = target.read_bytes()
            content_type = {".js": "text/javascript", ".css": "text/css", ".ttf": "font/ttf"}.get(
                target.suffix, "application/octet-stream")
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"Synthetic read-only citation UI fixture: http://127.0.0.1:{PORT}", flush=True)
    with ThreadingHTTPServer(("127.0.0.1", PORT), Handler) as server:
        server.serve_forever()
