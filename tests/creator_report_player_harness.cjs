// Offline event/DOM harness for the real report script; no browser or network.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
function element() {
  return { events: {}, addEventListener(name, fn) { this.events[name] = fn; } };
}
const player = Object.assign(element(), {
  currentSrc: '', readyState: 0, duration: NaN, currentTime: 0, plays: 0,
  removeAttribute() { this.currentSrc = ''; }, load() {},
  play() { this.plays++; return Promise.resolve(); },
});
const file = Object.assign(element(), {files: []});
const status = {textContent: ''};
const document = Object.assign(element(), {
  getElementById(id) { return {reportPlayer: player, reportMediaFile: file, reportPlayerStatus: status}[id]; },
});
const window = element();
const revoked = [];
let blobs = 0;
vm.runInNewContext(fs.readFileSync(process.argv[2], 'utf8'), {
  document, window, URL: {createObjectURL() { return 'blob:local-' + ++blobs; }, revokeObjectURL(url) {revoked.push(url);}},
});
function seek(value) { document.events.click({target: {closest() {return {dataset: {seek: value}};}}}); }
(async () => {
  seek('3'); assert.match(status.textContent, /请先选择/); assert.equal(player.plays, 0);
  file.files = [{type: 'application/json'}]; file.events.change(); assert.match(status.textContent, /不是可识别/);
  file.files = [{type: 'video/mp4'}]; file.events.change(); assert.equal(player.src, 'blob:local-1');
  player.currentSrc = player.src; player.readyState = 1; player.duration = 10;
  player.events.loadedmetadata(); assert.match(status.textContent, /未核验身份/);
  seek('5.125'); assert.equal(player.currentTime, 5.125); assert.equal(player.plays, 1);
  seek('11'); assert.match(status.textContent, /超过/); assert.equal(player.currentTime, 5.125);
  seek('NaN'); assert.match(status.textContent, /时间点无效/); assert.equal(player.plays, 1);
  player.play = () => Promise.reject(new Error('autoplay denied'));
  seek('2'); await Promise.resolve(); assert.equal(player.currentTime, 2); assert.match(status.textContent, /未自动播放/);
  player.events.error(); assert.match(status.textContent, /无法解码/);
  file.files = [{type: 'video/mp4'}]; file.events.change(); assert.deepEqual(revoked, ['blob:local-1']);
  player.currentSrc = player.src; player.readyState = 1; player.duration = Infinity;
  player.events.loadedmetadata(); assert.match(status.textContent, /无法读取/);
  window.events.pagehide(); assert.deepEqual(revoked, ['blob:local-1', 'blob:local-2']);
  file.files = []; file.events.change(); assert.match(status.textContent, /尚未选择/);
  console.log('player behavior assertions passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
