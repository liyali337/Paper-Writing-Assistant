import assert from "node:assert/strict";
import test from "node:test";

import { attachFigures, parseListItem, splitProseAndTables, splitReferenceEntries } from "./mdTable.ts";

test("splits lists, tables, and display math", () => {
  const blocks = splitProseAndTables(
    [
      "Contrastive training is the default [1, 2].",
      "",
      "Our contributions are:",
      "1. A hierarchical objective.",
      "2. A region aggregator.",
      "",
      "See Figure 1 for the overview.",
      "",
      "| Method | $mAP$ |",
      "|---|---|",
      "| Ours | 81.1 |",
      "",
      "Table 1. Zero-shot retrieval.",
    ].join("\n"),
  );
  assert.equal(blocks[0]?.type, "text");
  const list = blocks.find((block) => block.type === "list");
  assert.ok(list && list.type === "list" && list.ordered);
  assert.deepEqual(list.items, ["A hierarchical objective.", "A region aggregator."]);
  const table = blocks.find((block) => block.type === "table");
  assert.ok(table && table.type === "table");
  assert.equal(table.headers[1], "$mAP$");
  assert.equal(table.caption, "Table 1. Zero-shot retrieval.");
});

test("places figures next to the first mention", () => {
  const split = splitProseAndTables("The encoder is shown in Figure 1 and reused in Figure 2.");
  const { blocks, used } = attachFigures(split, [
    { figure_id: "fig-001", kind: "figure", label: "Figure 1", caption: "Overview." },
    { figure_id: "fig-002", kind: "figure", label: "Fig. 2", caption: "Pipeline." },
    { figure_id: "fig-003", kind: "formula", label: null, caption: null },
  ]);
  const ids = blocks.filter((block) => block.type === "figure").map((block) => block.figureId);
  assert.deepEqual(ids, ["fig-001", "fig-002"]);
  assert.ok(used.has("fig-001") && used.has("fig-002"));
  assert.equal(used.has("fig-003"), false);
});

test("does not treat 1.1 headings as list items", () => {
  assert.equal(parseListItem("1.1 Hierarchical Tokens"), null);
  assert.ok(parseListItem("1. A hierarchical objective."));
});

test("splits and strips bibliography markers", () => {
  const numbered = splitReferenceEntries(
    "[1] A. Radford et al. Learning transferable visual models.\n\n[2] C. Jia et al. Scaling up visual learning.",
  );
  assert.deepEqual(numbered, [
    "A. Radford et al. Learning transferable visual models.",
    "C. Jia et al. Scaling up visual learning.",
  ]);
  const glued = splitReferenceEntries(
    "A. Radford et al. Learning transferable visual models. [2] C. Jia et al. Scaling up visual learning.",
  );
  assert.equal(glued.length, 2);
  assert.match(glued[0] ?? "", /Radford/);
  assert.match(glued[1] ?? "", /Jia/);
  const lone = splitReferenceEntries("1\nA. Radford et al. Learning transferable visual models.");
  assert.equal(lone.length, 1);
  assert.match(lone[0] ?? "", /Radford/);
});
