// Exercise the actual editor script without writing any project review records.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    value: '', checked: false, disabled: false, textContent: '', style: {},
    children: [], options: [], selectedIndex: 0,
    append(child) { this.children.push(child); if (child.tag === 'option') { this.options.push(child); if (!this.value) this.value=String(child.value); } },
    replaceChildren() { this.children = []; this.options=[]; this.value=''; },
    querySelectorAll() { return this.children.flatMap(c => c.children || []).filter(c => c.tag === 'button'); },
  });
  return elements.get(id);
}
const canvas = element('canvas');
Object.assign(canvas, {
  width: 100, height: 100,
  getContext: () => new Proxy({}, {get: () => () => {}}),
  getBoundingClientRect: () => ({left: 0, top: 0, width: 100, height: 100}),
  setPointerCapture() {}, hasPointerCapture: () => false,
});
element('drawMode').value = 'positive';
element('factor').value = '4';
element('reviewer').value = 'human';
element('reason').value = 'complete landslide';
element('scope').value = 'full_landslide';
element('backgroundReason').value = 'entire region has no landslide';
element('target').value = 'A:1';
const posts = [];
let leaveAllowed = true;
const context = vm.createContext({
  document: {title:'',getElementById: element, querySelector: selector => element(selector), createElement: tag => ({tag, children: [], append(c) { this.children.push(c); }})},
  window: {}, confirm: () => leaveAllowed, URL,
  location: {href: 'http://127.0.0.1:8769/?batch=1'}, history: {replaceState() {}},
  fetch: async (url, opts) => {
    const payload = JSON.parse(opts.body); posts.push({url, payload});
    let result;
    if (url === '/api/background') result = {background: {background_id: 'BG_test', tile: payload.tile, aabb_px: payload.box, background_kind: payload.background_kind}};
    else if (url === '/api/save') result = {saved: {tile: payload.tile, source_id: payload.source_id}};
    else result = {removed: payload.background_id};
    return {ok: true, json: async () => result};
  },
});
const html = fs.readFileSync(path.join(__dirname, '../data/scripts/manual_box_review.html'), 'utf8');
const source = html.match(/<script>([\s\S]*?)<\/script>/)[1];
// Startup network loading is verified separately against the real local server.
vm.runInContext(source.slice(0, source.lastIndexOf('(async()=>')), context);
vm.runInContext("batch={items:[{tile:'A',source_id:1}],saved:{},excluded:{},backgrounds:{}};meta={tile:'A',source_id:1,window:[0,0,100,100],stride:1,polygons:[]};img={};loadedId='A:1';", context);
function draw(x0,y0,x1,y1) {
  canvas.onpointerdown({button: 0, pointerId: 1, clientX: x0, clientY: y0});
  canvas.onpointermove({clientX: x1, clientY: y1});
  canvas.onpointerup({pointerId: 1});
}
async function main() {
  draw(10,10,40,40);
  element('drawMode').value = 'background'; element('drawMode').onchange();
  assert.equal(element('save').disabled, true);
  assert.equal(element('saveBackground').disabled, false);
  draw(45,45,65,65);
  assert.equal(vm.runInContext('JSON.stringify(box)', context), '[10,10,40,40]');
  assert.equal(vm.runInContext('JSON.stringify(backgroundBox)', context), '[45,45,65,65]');
  await element('saveBackground').onclick();
  assert.equal(posts.length, 0); // cannot silently confirm a background
  element('backgroundConfirmed').checked = true;
  await element('saveBackground').onclick();
  assert.equal(posts[0].url, '/api/background');
  assert.equal('background_kind' in posts[0].payload, false);
  assert.deepEqual(posts[0].payload.box, [45,45,65,65]);
  assert.equal(vm.runInContext('dirty', context), true); // unsaved positive is preserved
  assert.equal(vm.runInContext('backgroundDirty', context), false);
  assert.equal(vm.runInContext('Object.keys(batch.saved).length', context), 0);
  element('drawMode').value = 'positive'; element('drawMode').onchange();
  element('confirmed').checked = true;
  await element('save').onclick();
  assert.equal(posts[1].url, '/api/save');
  assert.deepEqual(posts[1].payload.box, [10,10,40,40]);
  assert.equal(vm.runInContext('Object.keys(batch.backgrounds).length', context), 1);
  await vm.runInContext('removeBackground(batch.backgrounds.BG_test)', context);
  assert.equal(posts[2].url, '/api/background/remove');
  assert.equal(vm.runInContext('Object.keys(batch.backgrounds).length', context), 0);
  assert.equal(vm.runInContext('Object.keys(batch.saved).length', context), 1);
  vm.runInContext("batch.background_only=true;batch.groups=[{batch:1,items:[{tile:'A',source_id:1}]},{batch:2,items:[{tile:'B',source_id:2}]}];batch.items=batch.groups[0].items;batch.batch_number=1;batch.saved['B:2']={};load=async()=>{};populateTargets();updateProgress();setBusy(false);", context);
  assert.equal(element('save').disabled, true);
  assert.equal(element('exclude').disabled, true);
  assert.equal(element('drawMode').disabled, true);
  assert.ok(element('batchinfo').textContent.includes('已审核 1 个'));
  vm.runInContext('backgroundDirty=true', context);
  leaveAllowed = false; element('batchGroup').value='2'; element('batchGroup').onchange();
  assert.equal(vm.runInContext('batch.batch_number', context), 1);
  leaveAllowed = true; element('batchGroup').value='2'; element('batchGroup').onchange();
  assert.equal(vm.runInContext('batch.batch_number', context), 2);
  assert.equal(element('target').value, 'B:2');
  assert.ok(element('h1').textContent.includes('第2批'));
  console.log('Frontend verified: independent drawing, confirmation, save, progress, and undo.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
