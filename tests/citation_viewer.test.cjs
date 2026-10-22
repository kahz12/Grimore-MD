// Execute the actual unbundled controller without booting the application.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const script = fs.readFileSync(path.join(__dirname, '../grimore/api/static/app.js'), 'utf8');
const shared = 'Informe & cooperativa#p.3';
const details = [0, 1].map(i => ({
    id: `indexed:${i}`, label: shared, citation_ref: `cite:${i+1}`,
    document: { title: 'Informe & cooperativa', path: 'informe.pdf' },
    locator: { page: 3, chunk_index: i },
    passage: { text: i === 0 ? 'Distractor' : 'Evidencia esperada' }, source_state: 'modified',
}));

function selection(index, occurrence = 0) {
    return { citation_index: occurrence, label: shared, form: 'short',
        citation_ref: `cite:${index+1}`, source_detail_indices: [index], selection: 'exact' };
}

function controller() {
    const context = vm.createContext({ document: { addEventListener() {} }, console });
    vm.runInContext(script, context);
    return {
        run(code) { return vm.runInContext(code, context); },
        target(label = shared, occurrence = 0, sources = details, trace = [selection(1)]) {
            context.fixture = { label, occurrence, sources, trace };
            return JSON.parse(JSON.stringify(vm.runInContext(
                'citationTarget(fixture.label, fixture.occurrence, fixture.sources, fixture.trace)', context)));
        },
    };
}

test('explicit short citation selects the second record despite identical canonical labels', () => {
    assert.deepEqual(controller().target(), { kind: 'exact', indices: [1] });
});

test('different occurrences of one label retain different selected records', () => {
    const c = controller();
    const trace = [selection(1), selection(0, 1), selection(1, 2)];
    assert.deepEqual([0, 1, 2].map(i => c.target(shared, i, details, trace).indices), [[1], [0], [1]]);
});

test('shared canonical label asks for a choice, never first match', () => {
    const canonical = { ...selection(0), form: 'wikilink', citation_ref: null,
        source_detail_indices: [0, 1], selection: 'ambiguous' };
    const target = controller().target(shared, 0, details, [canonical]);
    assert.equal(target.kind, 'choose');
    assert.deepEqual(target.indices, [0, 1]);
});

test('legacy and source-list navigation can only automatically open a unique record', () => {
    const c = controller();
    assert.equal(c.target(shared, 0, details, null).kind, 'choose');
    assert.deepEqual(c.target(shared, 0, [details[1]], null).indices, [0]);
    assert.equal(c.target(shared, 0, [details[1]], null).kind, 'legacy');
    assert.equal(c.target(shared, null, details).kind, 'choose');
});

test('bare titles show all matching anchors; wrong anchors never match by title', () => {
    const c = controller();
    const sources = [details[0], { ...details[1], label: 'Informe & cooperativa#p.9' }];
    assert.deepEqual(c.target('Informe & cooperativa', null, sources, null).indices, [0, 1]);
    assert.deepEqual(c.target('Informe & cooperativa#p.99', null, sources, null).indices, []);
});

for (const [name, changes] of Object.entries({
    booleanIndex: { source_detail_indices: [true] }, negative: { source_detail_indices: [-1] },
    overflow: { source_detail_indices: [99] }, wrongRecord: { source_detail_indices: [0] },
    ambiguous: { source_detail_indices: [0, 1] }, wrongRef: { citation_ref: 'cite:99' },
    invalidRef: { citation_ref: 'cite:01' }, wrongLabel: { label: 'Wrong#p.3' },
    wrongOccurrence: { citation_index: 1 }, unknownForm: { form: 'invented' },
    falseCanonical: { form: 'wikilink', citation_ref: null },
})) {
    test(`inconsistent trace ${name} cannot silently select a passage`, () => {
        assert.equal(controller().target(shared, 0, details, [{ ...selection(1), ...changes }]).kind, 'choose');
    });
}

test('duplicate reference maps and duplicate trace occurrences are rejected', () => {
    const c = controller();
    assert.equal(c.target(shared, 0, details.map(d => ({ ...d, citation_ref: 'cite:2' }))).kind, 'choose');
    assert.equal(c.target(shared, 0, details, [selection(1), selection(0)]).kind, 'choose');
});

test('an explicitly empty trace is not mistaken for missing legacy data', () => {
    assert.equal(controller().target(shared, 0, [details[0]], []).kind, 'choose');
});

test('unique canonical selection does not need a short token', () => {
    const canonical = { ...selection(0), form: 'wikilink', citation_ref: null };
    assert.equal(controller().target(shared, 0, [details[0]], [canonical]).kind, 'exact');
});

test('rendered ordinals span paragraphs and repeated citations', () => {
    const c = controller();
    const html = c.run(`renderProse('Uno [[A]].\\n\\nDos [[A]] y [[B]].')`);
    assert.deepEqual([...html.matchAll(/data-citation-index="(\d+)"/g)].map(m => Number(m[1])), [0, 1, 2]);
});

test('HTML escaping preserves exact label identity, without double escaping', () => {
    const html = controller().run(`renderProse('<script>alert(1)</script> [[Informe & cooperativa#p.3]]')`);
    assert.ok(html.includes('data-label="Informe &amp; cooperativa#p.3"'));
    assert.ok(!html.includes('&amp;amp;'));
    assert.ok(!html.includes('<script>'));
    const hostile = controller().run(`renderCitedHtml('[[A" onmouseover="evil()]]')`);
    assert.ok(hostile.includes('data-label="A&quot; onmouseover=&quot;evil()"'));
});

test('newlines within a label do not break ordinal assignment', () => {
    const html = controller().run(`renderProse('[[A\\n\\nB]]\\n\\n[[C]]')`);
    assert.ok(html.includes('A&#10;&#10;B'));
    assert.deepEqual([...html.matchAll(/data-citation-index="(\d+)"/g)].map(m => m[1]), ['0', '1']);
});

test('clicks use occurrence-specific mapping; source list uses a chooser', () => {
    const c = controller();
    c.run(`state.sourceDetails = ${JSON.stringify(details)}; state.citationSelections = ${JSON.stringify([selection(1), selection(0, 1)])};
        var opened = []; openSourceDetail = i => opened.push(['detail', i]);
        openSourceChoices = (label, indices) => opened.push(['choices', indices]);
        openNoteByLabel(${JSON.stringify(shared)}, 0);
        openNoteByLabel(${JSON.stringify(shared)}, 1);
        openNoteByLabel(${JSON.stringify(shared)});`);
    assert.equal(c.run('JSON.stringify(opened)'), JSON.stringify([['detail', 1], ['detail', 0], ['choices', [0, 1]]]));
});

test('conversation reset closes the drawer and clears request-local references', () => {
    const c = controller();
    c.run(`var clearedDrawer = false;
        closeDrawer = () => { clearedDrawer = true; };
        renderConversation = () => {}; renderMarginalia = () => {}; hideRequestError = () => {};
        document.querySelector = () => ({focus() {}, appendChild() {}});
        document.createElement = () => ({});
        state.sourceDetails = ${JSON.stringify(details)};
        state.citationSelections = ${JSON.stringify([selection(1)])};
        resetConversation();`);
    assert.equal(c.run('clearedDrawer'), true);
    assert.equal(c.run('state.citationSelections'), null);
    assert.equal(c.run('state.sourceDetails.length'), 0);
});

test('rewiring after language changes is scoped and does not duplicate handlers', () => {
    const c = controller();
    c.run(`var listenerCount = 0;
        var citationElement = {dataset: {label: ${JSON.stringify(shared)}, citationIndex: '0'},
            setAttribute() {}, addEventListener() { listenerCount++; }};
        var answerBody = { querySelectorAll: () => [citationElement] };
        document.querySelector = selector => selector === '#answer-body' ? answerBody : null;
        document.querySelectorAll = () => { throw Error('Unrelated citations must not be rewired'); };
        state.sourceDetails = ${JSON.stringify(details)};
        state.citationSelections = ${JSON.stringify([selection(1)])};
        wireCitations(); wireCitations();`);
    assert.equal(c.run('listenerCount'), 2); // Exactly one click and one keyboard listener.
});
