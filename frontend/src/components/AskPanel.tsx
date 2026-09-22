import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";

import { askPaper, formatApiError } from "../api/client";
import type { AskTurn, Evidence, ExternalRef, Figure, IndexStatus, LibraryHit, PaperAnswer } from "../api/types";
import { demoAnswerFor, demoAskSuggestions } from "../data/demo";
import { FigureCard } from "./FigureViews";
import { ChatIcon, SendIcon } from "./icons";
import { MathText } from "./MathText";

export type AskMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  answer?: PaperAnswer;
};

type Props = {
  paperId: string | null;
  preview: boolean;
  ready: boolean;
  blockedReason?: string | null;
  indexHint?: string | null;
  indexStatus?: IndexStatus;
  figures: Figure[];
  messages: AskMessage[];
  onMessages: (updater: (current: AskMessage[]) => AskMessage[]) => void;
  onJump: (page: number, sectionId?: string) => void;
  onOpenFigure: (figure: Figure) => void;
  onRebuildIndex?: () => Promise<void> | void;
  onOpenPaper?: (paperId: string) => void;
};

function nextId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function toHistory(messages: AskMessage[]): AskTurn[] {
  return messages
    .filter((item) => item.text.trim())
    .map((item) => ({ role: item.role, content: item.text }));
}

async function previewAnswer(question: string): Promise<PaperAnswer> {
  await new Promise((resolve) => window.setTimeout(resolve, 450));
  return demoAnswerFor(question);
}

export function AskPanel({
  paperId,
  preview,
  ready,
  blockedReason,
  indexHint = null,
  indexStatus = preview ? "ready" : "pending",
  figures,
  messages,
  onMessages,
  onJump,
  onOpenFigure,
  onRebuildIndex,
  onOpenPaper,
}: Props) {
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const requestSeq = useRef(0);

  useEffect(() => {
    requestSeq.current += 1;
    setDraft("");
    setBusy(false);
  }, [paperId, preview]);

  useEffect(() => {
    const node = scroller.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [messages, busy]);

  const canAsk = ready && !busy && !blockedReason && (preview || Boolean(paperId));
  const suggestions = preview
    ? demoAskSuggestions
    : ["这篇在解决什么问题？", "核心方法是什么？", "我之前研读过类似方向的论文吗？", "在 arXiv 上检索相关论文"];

  async function ask(question: string) {
    const text = question.trim();
    if (!text || !canAsk) return;
    const history = toHistory(messages);
    const seq = requestSeq.current;
    const userMsg: AskMessage = { id: nextId(), role: "user", text };
    onMessages((current) => [...current, userMsg]);
    setDraft("");
    setBusy(true);
    try {
      let answer: PaperAnswer;
      if (preview) {
        answer = await previewAnswer(text);
      } else if (!paperId) {
        return;
      } else {
        answer = await askPaper(paperId, { question: text, history });
      }
      if (seq !== requestSeq.current) return;
      onMessages((current) => [
        ...current,
        { id: nextId(), role: "assistant", text: answer.answer_zh, answer },
      ]);
    } catch (error) {
      if (seq !== requestSeq.current) return;
      onMessages((current) => [
        ...current,
        { id: nextId(), role: "assistant", text: formatApiError(error) },
      ]);
    } finally {
      if (seq === requestSeq.current) {
        setBusy(false);
        inputRef.current?.focus();
      }
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    void ask(draft);
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void ask(draft);
    }
  }

  const indexChip =
    indexStatus === "ready"
      ? { label: preview ? "预览索引" : "索引就绪", tone: "ok" as const }
      : indexStatus === "failed"
        ? { label: "索引失败", tone: "bad" as const }
        : indexStatus === "skipped"
          ? { label: "未建索引", tone: "mute" as const }
          : { label: "索引中", tone: "warn" as const };

  return (
    <div className="ask-panel">
      <div className="ask-toolbar">
        <div className="ask-toolbar-title">
          <ChatIcon size={15} />
          <span>论文问答</span>
        </div>
        <span className={`chip is-${indexChip.tone}`}>
          <i />
          {indexChip.label}
        </span>
        {(indexStatus === "failed" || indexStatus === "pending") && onRebuildIndex ? (
          <button className="ghost-btn" type="button" onClick={() => void onRebuildIndex()}>
            {indexStatus === "failed" ? "重建索引" : "建立索引"}
          </button>
        ) : null}
      </div>

      <div className="ask-messages" ref={scroller}>
        {blockedReason ? <div className="callout is-bad">{blockedReason}</div> : null}
        {!blockedReason && indexHint ? <div className="callout is-info">{indexHint}</div> : null}

        {!messages.length && !busy ? (
          <div className="ask-empty">
            <h2>针对这篇论文提问</h2>
            <p>
              {preview
                ? "预览模式会返回示意回答与可跳转引用；正式版将按检索到的原文证据作答。"
                : "默认只根据当前这篇检索作答。问是否读过某方向会搜本地库；问检索 arXiv / 相关论文则走外部文献，不把网上条目当成这篇 PDF 的证据。"}
            </p>
            <div className="ask-suggestions">
              {suggestions.map((item) => (
                <button key={item} type="button" disabled={!canAsk} onClick={() => void ask(item)}>
                  {item}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {messages.map((message) =>
          message.role === "user" ? (
            <div key={message.id} className="ask-bubble is-user">
              {message.text}
            </div>
          ) : (
            <AssistantBubble
              key={message.id}
              message={message}
              figures={figures}
              paperId={paperId}
              preview={preview}
              onJump={onJump}
              onOpenFigure={onOpenFigure}
              onOpenPaper={onOpenPaper}
            />
          ),
        )}

        {busy ? (
          <div className="ask-bubble is-assistant is-pending">
            <span className="ask-typing" aria-hidden>
              <i />
              <i />
              <i />
            </span>
            正在检索…
          </div>
        ) : null}
      </div>

      <form className="ask-composer" onSubmit={onSubmit}>
        <textarea
          ref={inputRef}
          value={draft}
          rows={2}
          placeholder={blockedReason ? "当前无法提问" : "问这篇、本地库，或在 arXiv 上检索相关论文…"}
          disabled={!canAsk}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
        />
        <button className="primary-btn ask-send" type="submit" disabled={!canAsk || !draft.trim()}>
          <SendIcon />
          发送
        </button>
      </form>
    </div>
  );
}

function AssistantBubble({
  message,
  figures,
  paperId,
  preview,
  onJump,
  onOpenFigure,
  onOpenPaper,
}: {
  message: AskMessage;
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  onJump: (page: number, sectionId?: string) => void;
  onOpenFigure: (figure: Figure) => void;
  onOpenPaper?: (paperId: string) => void;
}) {
  const answer = message.answer;
  const related = (answer?.figure_ids || [])
    .map((id) => figures.find((figure) => figure.figure_id === id))
    .filter((figure): figure is Figure => Boolean(figure));
  const hits = (answer?.library_hits || []).filter((item) => item.paper_id !== paperId);
  const arxivMode = answer?.mode === "arxiv";

  return (
    <div className={`ask-bubble is-assistant${answer?.no_evidence || !answer ? " is-muted" : ""}`}>
      <div className="ask-answer">
        <MathText text={message.text} layout="ask" />
      </div>
      {answer?.citations?.length ? (
        <div className="ask-cites">
          {answer.citations.map((cite, index) => (
            <CitationChip key={`${cite.page}-${index}`} cite={cite} onJump={onJump} />
          ))}
        </div>
      ) : null}
      {hits.length ? (
        <div className="library-hits">
          <div className="ask-exts-label">{arxivMode ? "本地已有" : "本地库命中"}</div>
          {hits.map((hit) => (
            <LibraryHitCard key={hit.paper_id} hit={hit} onOpen={onOpenPaper} />
          ))}
        </div>
      ) : null}
      {answer?.external_refs?.length ? (
        <div className="ask-exts">
          <div className="ask-exts-label">{arxivMode ? "arXiv 命中" : "相关文献（外部）"}</div>
          {answer.external_refs.map((item, index) => (
            <ExternalRefChip key={`${item.identifier || item.title}-${index}`} item={item} />
          ))}
        </div>
      ) : null}
      {related.length && paperId ? (
        <div className="ask-figs">
          {related.map((figure) => (
            <FigureCard
              key={figure.figure_id}
              figure={figure}
              paperId={paperId}
              preview={preview}
              onOpen={onOpenFigure}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function LibraryHitCard({
  hit,
  onOpen,
}: {
  hit: LibraryHit;
  onOpen?: (paperId: string) => void;
}) {
  const title = hit.title?.trim() || hit.filename || hit.paper_id;
  const authors = (hit.authors || []).slice(0, 3).join("、");
  const openable = hit.openable !== false && Boolean(onOpen);
  return (
    <article className="library-hit">
      <div className="library-hit-body">
        <strong>{title}</strong>
        {authors ? <span className="library-hit-meta">{authors}</span> : null}
        {hit.abstract ? <p>{hit.abstract}</p> : null}
        {hit.why ? <span className="library-hit-why">{hit.why}</span> : null}
      </div>
      {onOpen ? (
        <button
          className="primary-btn library-hit-open"
          type="button"
          disabled={!openable}
          onClick={() => onOpen(hit.paper_id)}
        >
          {hit.openable === false ? "尚未就绪" : "进入这篇"}
        </button>
      ) : null}
    </article>
  );
}

function CitationChip({
  cite,
  onJump,
}: {
  cite: Evidence;
  onJump: (page: number, sectionId?: string) => void;
}) {
  const label = cite.section_title ? `${cite.section_title} · p.${cite.page}` : `p.${cite.page}`;
  return (
    <button
      className={`ask-cite${cite.sourced ? "" : " is-weak"}`}
      type="button"
      title={cite.quote}
      onClick={() => onJump(cite.page, cite.section_id ?? undefined)}
    >
      {label}
    </button>
  );
}

function ExternalRefChip({ item }: { item: ExternalRef }) {
  const year = item.year ? ` · ${item.year}` : "";
  const ident = item.identifier ? ` · ${item.identifier}` : "";
  const label = `${item.title}${year}${ident}`;
  if (item.url) {
    return (
      <a className="ask-ext" href={item.url} target="_blank" rel="noreferrer" title={item.snippet || item.title}>
        {label}
      </a>
    );
  }
  return (
    <span className="ask-ext is-plain" title={item.snippet || item.title}>
      {label}
    </span>
  );
}
