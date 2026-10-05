#!/usr/bin/env node
// Regenerates data/embeddings.bin + data/embeddings-manifest.json from
// data/statements.csv using the exact same model + pooling the browser uses
// for query embeddings (Xenova/all-MiniLM-L6-v2, mean pooling, normalized).
// Vectors ship int8-quantized: cosine similarity is scale-invariant per
// vector, so a fixed round(v*127) with no stored scale/zero-point is exact
// enough for ranking, and the client never needs to dequantize.
import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { parse } from "csv-parse/sync";
import { pipeline } from "@xenova/transformers";

// Long free-text fields get split into sentence-level units (better
// "why did this match" snippets); everything else embeds whole.
const LONG_FIELDS = new Set(["theAbstract", "theKeyAnalysis"]);

// Only these attributes are searchable text -- provenance/schema-definition
// fields (theProvenance, aComponent, anAuthor, theVenue, theDOI, ...) are not.
const EMBEDDED_FIELDS = new Set([
  "theName", "theAbstract", "theKeyAnalysis", "thePsychConstruct",
  "theAnalysisMethod", "theSampleParticipants", "theTrackingSetup",
  "anABCDomain", "aMotionFeature", "aTrackedSegment", "aTransformationApproach",
  "theIntention",
]);

function splitSentences(text) {
  const parts = text.split(/(?<=[.?!])\s+(?=[A-Z0-9"'(])/).map(s => s.trim()).filter(Boolean);
  return parts.length ? parts : [text.trim()];
}

const rows = parse(readFileSync("data/statements.csv", "utf8"), { columns: true });

// Entities with a theIntention row are lignin schema items (components/
// attributes), not research papers -- their theName (e.g. "aMotionFeature")
// is a schema label, not searchable title text, so it's excluded below.
const schemaEntities = new Set(
  rows.filter(r => r.attribute_name === "theIntention").map(r => r.entity)
);

const texts = [];
for (const r of rows) {
  if (!EMBEDDED_FIELDS.has(r.attribute_name)) continue;
  if (r.attribute_name === "theName" && schemaEntities.has(r.entity)) continue;
  if (!r.value) continue;
  if (LONG_FIELDS.has(r.attribute_name)) {
    for (const sentence of splitSentences(r.value)) {
      texts.push({ entity: r.entity, attribute_name: r.attribute_name, text: sentence });
    }
  } else {
    texts.push({ entity: r.entity, attribute_name: r.attribute_name, text: r.value });
  }
}

console.log(`Embedding ${texts.length} units with Xenova/all-MiniLM-L6-v2...`);
const extractor = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2");

const DIM = 384;
const manifestUnits = [];
const bin = Buffer.alloc(texts.length * DIM); // int8 per component, no header, fixed row width

for (const [i, t] of texts.entries()) {
  const out = await extractor(t.text, { pooling: "mean", normalize: true });
  const offset = i * DIM;
  for (let j = 0; j < DIM; j++) {
    const q = Math.round(out.data[j] * 127);
    bin.writeInt8(Math.max(-128, Math.min(127, q)), offset + j);
  }
  manifestUnits.push({ entity: t.entity, attribute_name: t.attribute_name, text: t.text });
  if (i % 100 === 0) console.log(`  ${i}/${texts.length}`);
}

mkdirSync("data", { recursive: true });
writeFileSync("data/embeddings.bin", bin);
writeFileSync(
  "data/embeddings-manifest.json",
  JSON.stringify({ model: "Xenova/all-MiniLM-L6-v2", dim: DIM, units: manifestUnits })
);
console.log(`Wrote data/embeddings.bin + manifest with ${manifestUnits.length} units.`);
