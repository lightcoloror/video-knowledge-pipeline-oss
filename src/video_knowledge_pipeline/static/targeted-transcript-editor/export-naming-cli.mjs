// export-naming-cli.ts
import { readFileSync, writeFileSync } from "node:fs";
import { resolve, join, basename } from "node:path";

// export-naming.ts
function safeExportPart(value, limit) {
  const cleaned = String(value || "").trim().replace(/[\u0000-\u001f\u007f<>:"/\\|?*\u202a-\u202e\u2066-\u2069]/g, "_").replace(/[. ]+$/g, "");
  if (!cleaned) throw Error("\u7F3A\u5C11\u9009\u9898\u6216\u89C6\u9891\u547D\u540D\u6807\u8BC6\uFF0C\u4E0D\u80FD\u5BFC\u51FA\u901A\u7528\u6587\u4EF6\u540D\u3002");
  return Array.from(cleaned).slice(0, limit).join("");
}
function projectClock(exported_at, context2) {
  if (!context2?.topic_title?.trim() || !context2?.video_title?.trim() || !context2?.video_id?.trim() || context2.time_zone !== "Asia/Shanghai") throw Error("\u7F3A\u5C11\u9009\u9898\u3001\u89C6\u9891\u6807\u8BC6\u6216\u5317\u4EAC\u65F6\u95F4\u8BBE\u7F6E\uFF1B\u8BF7\u5148\u914D\u7F6E\u5BFC\u51FA\u4E0A\u4E0B\u6587\u3002");
  const date = new Date(exported_at);
  if (!Number.isFinite(date.getTime())) throw Error("\u539F\u5BFC\u51FA\u65F6\u95F4\u65E0\u6548\uFF0C\u4E0D\u80FD\u7528\u5F53\u524D\u65F6\u95F4\u5192\u5145\u3002");
  const parts = Object.fromEntries(new Intl.DateTimeFormat("en-CA", { timeZone: context2.time_zone, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" }).formatToParts(date).map((p) => [p.type, p.value]));
  const ms = String(date.getUTCMilliseconds()).padStart(3, "0");
  const stamp = `${parts.year}${parts.month}${parts.day}_${parts.hour}${parts.minute}${parts.second}_${ms}_UTC+0800`;
  const local = `${parts.year}-${parts.month}-${parts.day}T${parts.hour}:${parts.minute}:${parts.second}.${ms}+08:00`;
  const identity = {
    schema: "vkp.review-export-identity.v1",
    topic_title: context2.topic_title,
    video_title: context2.video_title,
    video_id: context2.video_id,
    ...context2.pack_payload_sha256 ? { pack_payload_sha256: context2.pack_payload_sha256 } : {},
    exported_at_utc: date.toISOString(),
    exported_at_local: local,
    time_zone: context2.time_zone
  };
  return { prefix: [safeExportPart(context2.topic_title, 40), safeExportPart(context2.video_title, 24), safeExportPart(context2.video_id, 40)], stamp, identity };
}
function prepareIdentity(review, context2) {
  const clock = projectClock(review.exported_at, context2);
  if (!Array.isArray(review.chapters) || !review.chapters.length) throw Error("\u5BA1\u6838\u7ED3\u679C\u683C\u5F0F\u65E0\u6548\u3002");
  if (!context2?.source_hashes || !context2.source_files || new Set(review.chapters.map((c) => c.chapter)).size !== review.chapters.length || review.chapters.some((c) => context2.source_hashes[c.chapter] !== c.source_hash || context2.source_files[c.chapter] !== c.file) || Object.keys(context2.source_hashes).length !== review.chapters.length) throw Error("\u5BFC\u51FA\u547D\u540D\u4E0A\u4E0B\u6587\u4E0E\u5F53\u524D\u89C6\u9891\u6765\u6E90\u4E0D\u5339\u914D\u3002");
  const { stamp } = clock, identity = { ...clock.identity, scope: "all_chapters", chapter_count: review.chapters.length, source_files: review.chapters.map((c) => ({ chapter: c.chapter, file: c.file, source_hash: c.source_hash })) };
  const prefix = [...clock.prefix, `${review.chapters.length}\u6BB5\u53E3\u64AD`];
  return { prefix, stamp, identity };
}
function prepareReviewExport(review, context2) {
  if (review.schema !== "rwe.transcript-review.v3") throw Error("\u5BA1\u6838\u7ED3\u679C\u683C\u5F0F\u65E0\u6548\u3002");
  const { prefix, stamp, identity } = prepareIdentity(review, context2);
  const filename = [...prefix, "\u5BA1\u6838\u7ED3\u679C", "v3", "\u5BFC\u51FA" + stamp].join("_") + ".json";
  return { filename, payload: { ...review, export_identity: { ...identity, filename } } };
}
function prepareExactReviewExport(review, context2, chapters, requestIds) {
  if (!context2) throw Error("\u7F3A\u5C11\u9009\u9898\u3001\u89C6\u9891\u540D\u79F0\u6216\u89C6\u9891\u6807\u8BC6\uFF0C\u8BF7\u5148\u914D\u7F6E\u9879\u76EE\u8EAB\u4EFD\u3002");
  if (review.schema_version !== "vkp-targeted-review-decisions.draft.v1" || !context2 || review.pack_payload_sha256 !== context2.pack_payload_sha256) throw Error("\u7CBE\u786E\u5BA1\u6838\u7ED3\u679C\u4E0E\u9879\u76EE\u547D\u540D\u4E0A\u4E0B\u6587\u4E0D\u5339\u914D\u3002");
  if (!Array.isArray(review.decisions) || !review.decisions.length || !Array.isArray(requestIds) || new Set(requestIds).size !== requestIds.length || new Set(review.decisions.map((r) => r.review_id)).size !== review.decisions.length || review.decisions.some((r) => !requestIds.includes(r.review_id))) throw Error("\u7CBE\u786E\u5BA1\u6838\u9879\u6765\u6E90\u4E0D\u5339\u914D\uFF0C\u4E0D\u80FD\u5BFC\u51FA\u3002");
  if (!chapters.length) return prepareProjectReviewExport("\u7CBE\u786E\u5BA1\u6838\u7ED3\u679C", review, context2);
  const { prefix, stamp, identity } = prepareIdentity({ exported_at: review.exported_at, chapters }, context2);
  const filename = [...prefix, "\u7CBE\u786E\u5BA1\u6838\u7ED3\u679C", `${review.decisions.length}\u9879`, "\u5BFC\u51FA" + stamp].join("_") + ".json";
  return { filename, payload: { ...review, export_identity: { ...identity, scope: review.decisions.length === requestIds.length ? "all_requests" : "selected_requests", review_count: review.decisions.length, review_ids: review.decisions.map((r) => r.review_id), filename } } };
}
function prepareProjectReviewExport(kind, review, context2, locators = [], now = /* @__PURE__ */ new Date()) {
  const exported = review.exported_at || now.toISOString(), clock = projectClock(exported, context2);
  if (context2?.pack_payload_sha256 && review.pack_payload_sha256 && review.pack_payload_sha256 !== context2.pack_payload_sha256) throw Error("\u5BA1\u6838\u7ED3\u679C\u4E0E\u9879\u76EE\u6765\u6E90\u4E0D\u5339\u914D\u3002");
  const selected = review.decisions ?? review.reviews ?? review.rows ?? review.items ?? review.suggestions;
  const rows = Array.isArray(selected) ? selected : locators;
  const normalizedPath = (v) => String(v || "").replace(/\\/g, "/").replace(/\/$/, "").toLowerCase();
  const bundles = new Set(rows.map((r) => normalizedPath(r.bundle_dir)).filter(Boolean));
  if (bundles.size > 1 || context2?.bundle_dir && [...bundles].some((b) => b !== normalizedPath(context2.bundle_dir))) throw Error("\u5BA1\u6838\u5305\u542B\u5176\u4ED6\u9879\u76EE\uFF0C\u8BF7\u5206\u522B\u5BFC\u51FA\u6216\u5148\u914D\u7F6E\u6B63\u786E\u7684\u9879\u76EE\u8EAB\u4EFD\u3002");
  const seconds = (v) => {
    if (typeof v === "number") return Number.isFinite(v) && v >= 0 ? v : null;
    if (typeof v !== "string" || !v.trim()) return null;
    const p = v.trim().split(":");
    return p.every((x) => /^\d+(?:\.\d+)?$/.test(x)) ? p.reduce((n, x) => n * 60 + Number(x), 0) : null;
  };
  const interval = (r) => {
    let a = seconds(r.review_start_seconds ?? r.start_seconds ?? r.start ?? r.start_time), b = seconds(r.review_end_seconds ?? r.end_seconds ?? r.end ?? r.end_time);
    if ((a === null || b === null) && (r.time_range || r.time)) {
      const pair = String(r.time_range || r.time).split(/\s*[-–—]\s*/);
      if (pair.length === 2) [a, b] = pair.map(seconds);
    }
    return a !== null && b !== null && b >= a ? [a, b] : null;
  };
  const keys = ["review_id", "candidate_id", "section_id", "index", "timeline_index"];
  const spans = rows.map((row) => {
    if (interval(row)) return interval(row);
    const key = keys.find((k) => row[k] !== void 0 && row[k] !== null);
    const found = key && locators.find((r) => String(r[key] ?? (key === "timeline_index" ? r.index : void 0)) === String(row[key]) && (!row.bundle_dir || normalizedPath(r.bundle_dir) === normalizedPath(row.bundle_dir)));
    return found ? interval(found) : null;
  });
  let scope = rows.length ? `${rows.length}\u9879` : "0\u9879";
  const time = (s) => {
    const ms = Math.round(s * 1e3), pad = (v) => String(v).padStart(2, "0");
    return `${pad(Math.floor(ms / 36e5))}h${pad(Math.floor(ms / 6e4) % 60)}m${pad(Math.floor(ms / 1e3) % 60)}s${ms % 1e3 ? String(ms % 1e3).padStart(3, "0") : ""}`;
  };
  if (rows.length && spans.every(Boolean)) {
    const sorted = spans.sort((a, b) => a[0] - b[0]), merged = [];
    for (const s of sorted) {
      const last = merged.at(-1);
      if (last && s[0] <= last[1]) last[1] = Math.max(last[1], s[1]);
      else merged.push([...s]);
    }
    scope = merged.length === 1 ? `${time(merged[0][0])}-${time(merged[0][1])}` : `\u591A\u7247\u6BB5-${rows.length}\u9879`;
  }
  if (review.review_scope === "full_video") scope = "\u5168\u89C6\u9891";
  const filename = [...clock.prefix, safeExportPart(kind, 32), safeExportPart(scope, 64), "\u5BFC\u51FA" + clock.stamp].join("_") + ".json";
  const identity = { ...clock.identity, scope, review_count: rows.length, source_files: context2?.sources || [], review_ids: rows.map((r) => r.review_id ?? r.candidate_id ?? r.section_id ?? r.index ?? r.timeline_index).filter((v) => v !== void 0), filename };
  return { filename, payload: { ...review, exported_at: exported, export_identity: identity } };
}

// export-naming-cli.ts
var [source, contextFile, destination, packFile] = process.argv.slice(2);
if (!source || !contextFile || !destination) throw Error("Expected source JSON, naming context and existing output directory.");
var old = JSON.parse(readFileSync(source, "utf8"));
var context = JSON.parse(readFileSync(contextFile, "utf8"));
var result;
if (old.schema_version === "vkp-targeted-review-decisions.draft.v1") {
  if (!packFile) throw Error("Exact-review reissue requires the bound review pack.");
  const pack = JSON.parse(readFileSync(packFile, "utf8"));
  if (pack.pack_payload_sha256 !== old.pack_payload_sha256) throw Error("Source review pack does not match.");
  let exported = old.exported_at, provenance = "original_exported_at";
  if (!exported) {
    const match = basename(source).match(/__(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})(\d{3})(?:-\d+)?\.json$/);
    if (!match) throw Error("Historical export time is missing; refuse to replace it with now.");
    const [, y, m, d, h, minute, s, ms] = match;
    exported = (/* @__PURE__ */ new Date(`${y}-${m}-${d}T${h}:${minute}:${s}.${ms}+08:00`)).toISOString();
    provenance = "original_legacy_filename_fixed_UTC+0800";
  }
  result = prepareExactReviewExport({ ...old, exported_at: exported }, context, pack.transcript_edit.chapters.map((c) => ({ chapter: c.number, file: c.file, source_hash: c.source_text_hash })), pack.requests.map((r) => r.review_id));
  result.payload.export_identity.timestamp_provenance = provenance;
} else result = prepareReviewExport(old, context);
var target = join(resolve(destination), result.filename);
if (target === resolve(source)) throw Error("Refusing to overwrite the supplied export.");
writeFileSync(target, JSON.stringify(result.payload, null, 2) + "\n", { encoding: "utf8", flag: "wx" });
console.log(JSON.stringify({ path: target, exported_at: result.payload.export_identity.exported_at_utc, exported_at_local: result.payload.export_identity.exported_at_local, original_file_preserved: true }));
