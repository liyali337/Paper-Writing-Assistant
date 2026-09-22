import assert from "node:assert/strict";
import test from "node:test";

import { paperAskAccess } from "./askGate.ts";

test("preview always opens the composer", () => {
  const gate = paperAskAccess({ preview: true, waiting: true, status: "parsing" });
  assert.equal(gate.canOpenComposer, true);
  assert.equal(gate.closeReadAvailable, true);
  assert.equal(gate.blockedReason, null);
});

test("parsing blocks every mode", () => {
  const gate = paperAskAccess({ preview: false, waiting: true, status: "parsing" });
  assert.equal(gate.canOpenComposer, false);
  assert.match(gate.blockedReason ?? "", /解析/);
});

test("index pending still allows library and arxiv", () => {
  const gate = paperAskAccess({
    preview: false,
    waiting: false,
    status: "ready",
    indexStatus: "pending",
  });
  assert.equal(gate.canOpenComposer, true);
  assert.equal(gate.closeReadAvailable, false);
  assert.equal(gate.blockedReason, null);
  assert.match(gate.indexHint ?? "", /本地库/);
});

test("index failed does not lock the input", () => {
  const gate = paperAskAccess({
    preview: false,
    waiting: false,
    status: "ready",
    indexStatus: "failed",
    indexError: "qdrant down",
  });
  assert.equal(gate.canOpenComposer, true);
  assert.equal(gate.closeReadAvailable, false);
  assert.equal(gate.indexHintTone, "bad");
  assert.match(gate.indexHint ?? "", /qdrant down/);
});

test("index skipped still allows library and arxiv", () => {
  const gate = paperAskAccess({
    preview: false,
    waiting: false,
    status: "ready",
    indexStatus: "skipped",
  });
  assert.equal(gate.canOpenComposer, true);
  assert.equal(gate.closeReadAvailable, false);
  assert.match(gate.indexHint ?? "", /arXiv/);
});

test("index ready unlocks close read", () => {
  const gate = paperAskAccess({
    preview: false,
    waiting: false,
    status: "ready",
    indexStatus: "ready",
  });
  assert.equal(gate.canOpenComposer, true);
  assert.equal(gate.closeReadAvailable, true);
  assert.equal(gate.indexHint, null);
});
