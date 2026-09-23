import assert from "node:assert/strict";
import test from "node:test";

import { applyToolStep, liveHeadline, parseSseBlocks, type AskToolEvent } from "./askStream.ts";

test("parseSseBlocks keeps a partial frame", () => {
  const first = parseSseBlocks('event: phase\ndata: {"type":"phase","phase":"dialogue","label":"正在理解问题"}\n\nevent: tool\ndata: {"type":');
  assert.equal(first.events.length, 1);
  assert.equal(first.events[0].type, "phase");
  assert.match(first.rest, /event: tool/);

  const second = parseSseBlocks(
    `${first.rest}"tool","status":"start","label":"检索文内片段","call_id":"c1","detail":"lambda","note":""}\n\n`,
  );
  assert.equal(second.events.length, 1);
  assert.equal(second.events[0].type, "tool");
  assert.equal(second.rest, "");
});

test("applyToolStep updates the same call and headline follows the running tool", () => {
  const start: AskToolEvent = {
    type: "tool",
    name: "search_child_chunks",
    status: "start",
    label: "检索文内片段",
    call_id: "c1",
    detail: "lambda",
    note: "",
  };
  const running = applyToolStep([], start);
  assert.equal(liveHeadline("正在理解问题", running), "正在检索文内片段");

  const done: AskToolEvent = { ...start, status: "done", note: "命中 2 条", detail: "" };
  const finished = applyToolStep(running, done);
  assert.equal(finished.length, 1);
  assert.equal(finished[0].status, "done");
  assert.equal(finished[0].detail, "lambda");
  assert.equal(finished[0].note, "命中 2 条");
  assert.equal(liveHeadline("正在整理回答", finished), "正在整理回答");
});
