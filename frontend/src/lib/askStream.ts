import type { PaperAnswer } from "../api/types";

export type AskPhaseEvent = {
  type: "phase";
  phase: string;
  label: string;
};

export type AskToolEvent = {
  type: "tool";
  name: string;
  status: "start" | "done" | "skip";
  label: string;
  call_id: string;
  detail: string;
  note: string;
};

export type AskAnswerEvent = {
  type: "answer";
  answer: PaperAnswer;
};

export type AskErrorEvent = {
  type: "error";
  status: number;
  code: string;
  stage: string;
  message: string;
};

export type AskStreamEvent = AskPhaseEvent | AskToolEvent | AskAnswerEvent | AskErrorEvent;

export type AskLiveStep = {
  id: string;
  label: string;
  detail: string;
  note: string;
  status: "start" | "done" | "skip";
};

export function parseSseBlocks(buffer: string): { events: AskStreamEvent[]; rest: string } {
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  const events: AskStreamEvent[] = [];
  for (const block of parts) {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n")
      .trim();
    if (!data) continue;
    const parsed = JSON.parse(data) as AskStreamEvent;
    if (parsed && typeof parsed === "object" && typeof parsed.type === "string") {
      events.push(parsed);
    }
  }
  return { events, rest };
}

export function applyToolStep(steps: AskLiveStep[], event: AskToolEvent): AskLiveStep[] {
  const id = event.call_id || `${event.name}-${steps.length}`;
  const next: AskLiveStep = {
    id,
    label: event.label || event.name,
    detail: event.detail || "",
    note: event.note || "",
    status: event.status,
  };
  const index = steps.findIndex((item) => item.id === id);
  if (index < 0) return [...steps, next];
  const copy = steps.slice();
  const previous = copy[index];
  copy[index] = {
    ...previous,
    ...next,
    detail: next.detail || previous.detail,
  };
  return copy;
}

export function liveHeadline(label: string, steps: AskLiveStep[]): string {
  const running = [...steps].reverse().find((item) => item.status === "start");
  if (running) return `正在${running.label}`;
  return label || "正在检索…";
}
