import assert from "node:assert/strict";
import test from "node:test";

import { attachFigures, collectLabeledTables, layoutPaperMedia, normalizeTableLabel, parseListItem, splitProseAndTables, splitReferenceEntries } from "./mdTable.ts";

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

test("keeps roman TABLE I captions and dagger notes on the table", () => {
  const blocks = splitProseAndTables(
    [
      "| Methods | mAP |",
      "|---|---|",
      "| Ours | 81.1 |",
      "",
      "TABLE I COMPARISON OF DIFFERENT METHODS ON PEDESTRIAN DATASET RGBNT201. († DENOTES THE METHODS THAT ADDITIONALLY UTILIZE TEXT ANNOTATIONS FROM IDEA [30].)",
    ].join("\n"),
  );
  const table = blocks.find((block) => block.type === "table");
  assert.ok(table && table.type === "table");
  assert.equal(blocks.filter((block) => block.type === "text").length, 0);
  assert.match(table.caption ?? "", /^TABLE I COMPARISON/);
  assert.match(table.note ?? "", /DENOTES THE METHODS/);
  assert.equal(normalizeTableLabel(table.caption), "table 1");
});

test("glues a caption split across a markdown table back onto that table", () => {
  const blocks = splitProseAndTables(
    [
      "TABLE III PERFORMANCE OF MISSING-MODALITY SETTINGS ON RGBNT $201.(\\ast$ INDICATES THAT THE MASA IS NOT APPLIED; INSTEAD, FULLY DECOUPLED",
      "| Methods | mAP |",
      "|---|---|",
      "| Ours | 42.4 |",
      "",
      "FEATURES ARE CONCATENATED AS THE FINAL REPRESENTATION.) 'M (X)' MEANS MISSING THE X IMAGE MODALITY.",
      "",
      "Modality Mismatch Protocol. We further evaluate mismatched sensors.",
    ].join("\n"),
  );
  const table = blocks.find((block) => block.type === "table");
  assert.ok(table && table.type === "table");
  assert.match(table.caption ?? "", /TABLE III PERFORMANCE/);
  assert.match(table.note ?? "", /CONCATENATED/);
  assert.match(table.note ?? "", /M \(X\)/);
  const prose = blocks.filter((block) => block.type === "text").map((block) => block.text);
  assert.equal(prose.length, 1);
  assert.match(prose[0] ?? "", /Modality Mismatch Protocol/);
});

test("moves TABLE I to the first Table I mention and keeps the note attached", () => {
  const layout = layoutPaperMedia(
    [
      {
        section_id: "sec-impl",
        text: [
          "Training uses Adam.",
          "",
          "| Methods | mAP |",
          "|---|---|",
          "| Ours | 81.1 |",
          "",
          "TABLE I COMPARISON OF DIFFERENT METHODS. († DENOTES TEXT ANNOTATIONS.)",
        ].join("\n"),
        figure_ids: [],
      },
      {
        section_id: "sec-sota",
        text: "Table I compares pedestrian datasets.",
        figure_ids: [],
      },
    ],
    [],
  );
  const impl = layout.get("sec-impl") ?? [];
  const sota = layout.get("sec-sota") ?? [];
  assert.equal(impl.filter((block) => block.type === "table").length, 0);
  assert.equal(
    impl.filter((block) => block.type === "text" && /DENOTES/.test(block.text)).length,
    0,
  );
  const table = sota.find((block) => block.type === "table");
  assert.ok(table && table.type === "table");
  assert.match(table.caption ?? "", /TABLE I COMPARISON/);
  assert.match(table.note ?? "", /DENOTES TEXT ANNOTATIONS/);
});

test("adopts an all-caps orphan title onto a TABLE IV stub caption", () => {
  const blocks = splitProseAndTables(
    [
      "ly consis- PERFORMANCE OF MODALITY-MISMATCHED SCENARIOS.(∗ INDICATES THAT THE MASA IS NOT APPLIED; INSTEAD, FULLY DECOUPLED FEATURES ARE CONCATENATED AS THE FINAL REPRESENTATION.)",
      "",
      "| Methods | mAP |",
      "|---|---|",
      "| HAMNet | 27.7 |",
      "",
      "TABLE II COMPARISON ON VEHICLE DATASETS.",
      "",
      "TABLE IV",
      "",
      "| Methods | Avg |",
      "|---|---|",
      "| Ours | 54.9 |",
    ].join("\n"),
  );
  const tables = blocks.filter((block) => block.type === "table");
  assert.equal(tables.length, 2);
  assert.ok(tables[0] && tables[0].type === "table");
  assert.match(tables[0].caption ?? "", /TABLE II COMPARISON/);
  assert.ok(tables[1] && tables[1].type === "table");
  assert.match(tables[1].caption ?? "", /TABLE IV/);
  assert.match(tables[1].caption ?? "", /MISMATCHED SCENARIOS/);
  assert.match(tables[1].note ?? "", /MASA IS NOT APPLIED/);
  assert.equal(
    blocks.filter((block) => block.type === "text" && /MISMATCHED/.test(block.text)).length,
    0,
  );
});

test("peels a table title that PDF glued onto the previous sentence", () => {
  const blocks = splitProseAndTables(
    [
      "Matching is performed on structurally consis- PERFORMANCE OF MODALITY-MISMATCHED SCENARIOS.(∗ INDICATES THAT THE MASA IS NOT APPLIED; INSTEAD, FULLY DECOUPLED FEATURES ARE CONCATENATED AS THE FINAL REPRESENTATION.)",
      "",
      "| Methods | Avg |",
      "|---|---|",
      "| Ours | 54.9 |",
      "",
      "TABLE IV",
      "",
      "Modality Mismatch Protocol. We further evaluate mismatched sensors.",
    ].join("\n"),
  );
  const table = blocks.find((block) => block.type === "table");
  assert.ok(table && table.type === "table");
  assert.match(table.caption ?? "", /TABLE IV/);
  assert.match(table.caption ?? "", /MISMATCHED SCENARIOS/);
  assert.match(table.note ?? "", /MASA IS NOT APPLIED/);
  const prose = blocks.filter((block) => block.type === "text").map((block) => block.text);
  assert.match(prose[0] ?? "", /Matching is performed/);
  assert.doesNotMatch(prose[0] ?? "", /MISMATCHED SCENARIOS/);
  assert.match(prose.at(-1) ?? "", /Modality Mismatch Protocol/);
});

test("does not treat a Table I discussion sentence as a caption", () => {
  const blocks = splitProseAndTables("Table I compares pedestrian datasets on RGBNT201.");
  assert.equal(blocks.length, 1);
  assert.equal(blocks[0]?.type, "text");
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

test("keeps display-math fences on their own lines", () => {
  const blocks = splitProseAndTables(
    [
      "formally expressed as:",
      "",
      "$$",
      "Q_T^T = \\Phi(X), \\tag{2}",
      "$$ where $\\Phi$ denotes the embedding.",
    ].join("\n"),
  );
  const math = blocks.find((block) => block.type === "text" && block.text.includes("$$"));
  assert.ok(math && math.type === "text");
  assert.match(math.text, /^\$\$\n/);
  assert.match(math.text, /\\tag\{2\}\n\$\$ where/);
});

test("drops standalone figure captions that already live on the image", () => {
  const blocks = splitProseAndTables("Figure 1. Overview of the proposed method.");
  assert.equal(blocks.length, 0);
});

test("keeps sentences that start by citing a figure", () => {
  const blocks = splitProseAndTables(
    "Fig. 3 illustrates generation. Fig. 4 presents the modules.",
  );
  assert.equal(blocks.length, 1);
  assert.equal(blocks[0]?.type, "text");
  assert.match(blocks[0] && blocks[0].type === "text" ? blocks[0].text : "", /Fig\. 3 illustrates/);
});

test("drops standalone Chinese figure captions that already live on the image", () => {
  const blocks = splitProseAndTables("图 1. 方法总览：三个视觉层级对齐。");
  assert.equal(blocks.length, 0);
});

test("keeps Chinese sentences that start by citing a figure", () => {
  const blocks = splitProseAndTables("图 3 展示了生成过程。图 4 给出了模块结构。");
  assert.equal(blocks.length, 1);
  assert.equal(blocks[0]?.type, "text");
  assert.match(blocks[0] && blocks[0].type === "text" ? blocks[0].text : "", /图 3 展示了/);
});

test("splits stacked markdown tables that share no blank line", () => {
  const blocks = splitProseAndTables(
    [
      "| Method | mAP |",
      "|---|---|",
      "| Ours | 81.1 |",
      "| Dataset | Acc |",
      "|---|---|",
      "| COCO | 90 |",
    ].join("\n"),
  );
  const tables = blocks.filter((block) => block.type === "table");
  assert.equal(tables.length, 2);
  assert.ok(tables[0] && tables[0].type === "table");
  assert.equal(tables[0].headers[0], "Method");
  assert.ok(tables[1] && tables[1].type === "table");
  assert.equal(tables[1].headers[0], "Dataset");
});

test("moves dumped tables to the first Table N mention and keeps them unclustered", () => {
  const layout = layoutPaperMedia(
    [
      {
        section_id: "sec-impl",
        text: [
          "Training uses Adam.",
          "",
          "| Method | mAP |",
          "|---|---|",
          "| Ours | 81.1 |",
          "",
          "Table 1. Vehicle results.",
        ].join("\n"),
        figure_ids: [],
      },
      {
        section_id: "sec-sota",
        text: "Table 1 compares vehicle datasets.\n\nTable 2 outlines person results.",
        figure_ids: [],
      },
      {
        section_id: "sec-ablate",
        text: [
          "Table 5 shows module ablations.",
          "",
          "| Methods | mAP |",
          "|---|---|",
          "| HAMNet | 27.7 |",
          "",
          "Table 2. Person results.",
          "",
          "| Index | CRE | mAP |",
          "|---|---|---|",
          "| A | x | 70.1 |",
          "",
          "Table 5. Module ablation.",
        ].join("\n"),
        figure_ids: [],
      },
    ],
    [],
  );
  const impl = layout.get("sec-impl") ?? [];
  const sota = layout.get("sec-sota") ?? [];
  const ablate = layout.get("sec-ablate") ?? [];
  assert.equal(
    impl.filter((block) => block.type === "table").length,
    0,
  );
  const sotaTables = sota.filter((block) => block.type === "table");
  assert.equal(sotaTables.length, 2);
  assert.ok(sotaTables[0] && sotaTables[0].type === "table");
  assert.match(sotaTables[0].caption ?? "", /^Table 1/);
  assert.ok(sotaTables[1] && sotaTables[1].type === "table");
  assert.match(sotaTables[1].caption ?? "", /^Table 2/);
  const sotaTypes = sota.map((block) => block.type);
  assert.deepEqual(sotaTypes, ["text", "table", "text", "table"]);
  const ablateTables = ablate.filter((block) => block.type === "table");
  assert.equal(ablateTables.length, 1);
  assert.ok(ablateTables[0] && ablateTables[0].type === "table");
  assert.match(ablateTables[0].caption ?? "", /^Table 5/);
  assert.equal(ablate[0]?.type, "text");
  assert.equal(ablate[1]?.type, "table");
});

test("shows a duplicated Table 1 markdown only once, at the first mention", () => {
  const layout = layoutPaperMedia(
    [
      {
        section_id: "sec-impl",
        text: [
          "Training uses Adam.",
          "",
          "| Method | mAP |",
          "|---|---|",
          "| Ours | 81.1 |",
          "",
          "Table 1. Vehicle results.",
        ].join("\n"),
        figure_ids: [],
      },
      {
        section_id: "sec-sota",
        text: "Table 1 compares vehicle datasets.",
        figure_ids: [],
      },
      {
        section_id: "sec-dump",
        text: [
          "Appendix numbers.",
          "",
          "| Method | mAP |",
          "|---|---|",
          "| Ours | 81.1 |",
          "",
          "Table 1. Vehicle results.",
        ].join("\n"),
        figure_ids: [],
      },
    ],
    [],
  );
  const tables = [...layout.values()]
    .flat()
    .filter((block) => block.type === "table");
  assert.equal(tables.length, 1);
  assert.equal(
    (layout.get("sec-sota") ?? []).filter((block) => block.type === "table").length,
    1,
  );
  assert.equal(
    (layout.get("sec-impl") ?? []).filter((block) => block.type === "table").length,
    0,
  );
  assert.equal(
    (layout.get("sec-dump") ?? []).filter((block) => block.type === "table").length,
    0,
  );
});

test("places an unused Table 4 at a mismatched Table 10 mention", () => {
  const layout = layoutPaperMedia(
    [
      {
        section_id: "sec-ablate",
        text: [
          "Table 5 shows module ablations.",
          "",
          "| Index | CRE | mAP |",
          "|---|---|---|",
          "| A | x | 70.1 |",
          "",
          "Table 5. Module ablation.",
          "",
          "| Methods | mAP |",
          "|---|---|",
          "| 3M Loss | 72.2 |",
          "",
          "Table 4. CT-CMC versus 3M Loss.",
          "",
          "Discussion of CT-CMC. Table 10 validates the cross-modal constraint.",
        ].join("\n"),
        figure_ids: [],
      },
    ],
    [],
  );
  const blocks = layout.get("sec-ablate") ?? [];
  const tables = blocks.filter((block) => block.type === "table");
  assert.equal(tables.length, 2);
  assert.ok(tables[0] && tables[0].type === "table");
  assert.match(tables[0].caption ?? "", /^Table 5/);
  assert.ok(tables[1] && tables[1].type === "table");
  assert.match(tables[1].caption ?? "", /^Table 4/);
  const types = blocks.map((block) =>
    block.type === "table" ? `table:${block.caption?.slice(0, 8)}` : block.type,
  );
  assert.deepEqual(types, [
    "text",
    "table:Table 5.",
    "text",
    "table:Table 4.",
  ]);
});

test("shows each figure once at the first mention, not again as a section leftover", () => {
  const layout = layoutPaperMedia(
    [
      {
        section_id: "sec-related",
        text: "Prior CoT work is reviewed here.",
        figure_ids: ["fig-003"],
      },
      {
        section_id: "sec-method",
        text: "Fig. 3 illustrates generation. Fig. 4 presents the modules.",
        figure_ids: [],
      },
      {
        section_id: "sec-caption",
        text: "As shown in Fig. 3, each modality has its own template.",
        figure_ids: ["fig-004"],
      },
    ],
    [
      {
        figure_id: "fig-003",
        kind: "figure",
        label: "Figure 3",
        caption: "Figure 3. Templates.",
        section_id: "sec-related",
      },
      {
        figure_id: "fig-004",
        kind: "figure",
        label: "Figure 4",
        caption: "Figure 4. Framework.",
        section_id: "sec-caption",
      },
    ],
  );
  const ids = [...layout.values()]
    .flat()
    .filter((block) => block.type === "figure")
    .map((block) => (block.type === "figure" ? block.figureId : ""));
  assert.deepEqual(ids, ["fig-003", "fig-004"]);
  const method = layout.get("sec-method") ?? [];
  assert.deepEqual(
    method.filter((block) => block.type === "figure").map((block) => (block.type === "figure" ? block.figureId : "")),
    ["fig-003", "fig-004"],
  );
  assert.equal(
    (layout.get("sec-related") ?? []).filter((block) => block.type === "figure").length,
    0,
  );
  assert.equal(
    (layout.get("sec-caption") ?? []).filter((block) => block.type === "figure").length,
    0,
  );
});

test("does not treat 1.1 headings as list items", () => {
  assert.equal(parseListItem("1.1 Hierarchical Tokens"), null);
  assert.ok(parseListItem("1. A hierarchical objective."));
});

test("collectLabeledTables keeps the first captioned table per label", () => {
  const map = collectLabeledTables([
    "| A | B |\n|---|---|\n| 1 | 2 |\n\n表 1. 结果。",
    "| A | B |\n|---|---|\n| 9 | 9 |\n\n表 1. 重复。",
  ]);
  assert.equal(map.size, 1);
  assert.match(map.get("table 1")?.caption ?? "", /结果/);
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
