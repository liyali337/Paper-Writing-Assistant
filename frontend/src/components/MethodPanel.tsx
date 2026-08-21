import type { Figure, MethodExplain, MethodStep, Section } from "../api/types";
import { FigureCard } from "./FigureViews";
import { SectionReader, methodLikeSections } from "./SectionReader";

type Props = {
  method: MethodExplain | null;
  sections?: Section[];
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  focusSectionId?: string | null;
  onJump: (page: number) => void;
  onOpenFigure: (figure: Figure) => void;
};

export function MethodPanel({
  method,
  sections = [],
  figures,
  paperId,
  preview,
  focusSectionId,
  onJump,
  onOpenFigure,
}: Props) {
  if (!method) {
    return (
      <SectionReader
        sections={methodLikeSections(sections)}
        figures={figures}
        paperId={paperId}
        preview={preview}
        heading="方法原文"
        note="中文方法详解将在下一阶段生成。下面是 Method / Experiments 各节原文与图。"
        focusId={focusSectionId}
        onJump={onJump}
        onOpenFigure={onOpenFigure}
      />
    );
  }

  const unused = method.figure_ids.filter(
    (id) =>
      !method.pipeline_steps.some((step) => step.figure_ids.includes(id)) &&
      !method.key_modules.some((step) => step.figure_ids.includes(id)),
  );

  return (
    <article className="prose">
      {method.partial ? (
        <div className="callout">方法为部分生成（过长或超预算），可稍后刷新。</div>
      ) : null}

      <h2>方法怎么走</h2>
      <p className="lead">{method.overview_zh}</p>

      {unused.map((id) => {
        const figure = figures.find((item) => item.figure_id === id);
        return figure ? (
          <FigureCard
            key={id}
            figure={figure}
            paperId={paperId}
            preview={preview}
            onOpen={onOpenFigure}
          />
        ) : null;
      })}

      <section className="block">
        <h3>流水线</h3>
        {method.pipeline_steps.map((step, index) => (
          <Step
            key={`${step.name}-${index}`}
            index={index + 1}
            step={step}
            figures={figures}
            paperId={paperId}
            preview={preview}
            onJump={onJump}
            onOpenFigure={onOpenFigure}
          />
        ))}
      </section>

      {method.key_modules.length ? (
        <section className="block">
          <h3>关键模块</h3>
          {method.key_modules.map((step, index) => (
            <Step
              key={`${step.name}-m-${index}`}
              index={index + 1}
              step={step}
              figures={figures}
              paperId={paperId}
              preview={preview}
              onJump={onJump}
              onOpenFigure={onOpenFigure}
            />
          ))}
        </section>
      ) : null}

      {method.vs_prior_zh ? (
        <section className="block">
          <h3>相对旧方案</h3>
          <p>{method.vs_prior_zh}</p>
        </section>
      ) : null}

      {method.assumptions_zh.length ? (
        <section className="block">
          <h3>假设与局限</h3>
          <ul>
            {method.assumptions_zh.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      ) : null}

      {method.open_to_read_zh.length ? (
        <section className="block">
          <h3>建议对照原文</h3>
          <ul>
            {method.open_to_read_zh.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </section>
      ) : null}
    </article>
  );
}

function Step({
  index,
  step,
  figures,
  paperId,
  preview,
  onJump,
  onOpenFigure,
}: {
  index: number;
  step: MethodStep;
  figures: Figure[];
  paperId: string | null;
  preview: boolean;
  onJump: (page: number) => void;
  onOpenFigure: (figure: Figure) => void;
}) {
  return (
    <div className="step">
      <div className="step-head">
        <span className="step-n">{String(index).padStart(2, "0")}</span>
        <h4>{step.name}</h4>
      </div>
      <p>{step.what_zh}</p>
      <div className="step-meta">
        <button className="cite" type="button" onClick={() => onJump(step.page)}>
          p.{step.page}
        </button>
        <span>{step.section_title}</span>
      </div>
      {step.quote ? (
        <details className="quote">
          <summary>{step.sourced ? "原文摘录 · 已核对" : "原文摘录 · 未核对"}</summary>
          <blockquote>{step.quote}</blockquote>
        </details>
      ) : null}
      {step.figure_ids.map((id) => {
        const figure = figures.find((item) => item.figure_id === id);
        return figure ? (
          <FigureCard
            key={id}
            figure={figure}
            paperId={paperId}
            preview={preview}
            onOpen={onOpenFigure}
          />
        ) : null;
      })}
    </div>
  );
}
