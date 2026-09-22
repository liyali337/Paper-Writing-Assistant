import type {
  AskRequest,
  Evidence,
  Figure,
  Health,
  LibraryHit,
  MethodExplain,
  Paper,
  PaperAnswer,
  PaperIntro,
  PaperTranslation,
  Section,
} from "./types";
import { HttpError, type ApiError } from "./types";

export type { Figure, Health, MethodExplain, Paper, PaperIntro, PaperTranslation, Section };
export { HttpError };

async function request<T>(path: string, init?: RequestInit, timeoutMs = 15000): Promise<T> {
  const signal = init?.signal ?? AbortSignal.timeout(timeoutMs);
  const response = await fetch(`/api${path}`, { ...init, signal });
  const payload = await response.json().catch(() => null);
  if (response.status === 202) {
    throw new HttpError(202, "pending");
  }
  if (!response.ok) {
    const detail = payload?.detail as ApiError | undefined;
    throw new HttpError(
      response.status,
      detail?.message || `HTTP ${response.status}`,
      detail && typeof detail === "object" ? detail : undefined,
    );
  }
  return payload as T;
}

export function getHealth() {
  return request<Health>("/health");
}

export function listPapers() {
  return request<Paper[]>("/papers", undefined, 8000);
}

export function uploadPaper(file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<{ paper_id: string; status: string }>("/papers", { method: "POST", body }, 120000);
}

export function getPaper(paperId: string) {
  return request<Paper>(`/papers/${paperId}`);
}

export function getSections(paperId: string) {
  return request<Section[]>(`/papers/${paperId}/sections`);
}

export function getFigures(paperId: string, sectionId?: string) {
  const query = sectionId ? `?section_id=${encodeURIComponent(sectionId)}` : "";
  return request<Figure[]>(`/papers/${paperId}/figures${query}`);
}

export function getIntro(paperId: string) {
  return request<PaperIntro>(`/papers/${paperId}/intro`);
}

export function getMethod(paperId: string) {
  return request<MethodExplain>(`/papers/${paperId}/method`);
}

export function refreshIntro(paperId: string) {
  return request<PaperIntro>(`/papers/${paperId}/intro?refresh=true`, { method: "POST" });
}

export function refreshMethod(paperId: string) {
  return request<MethodExplain>(`/papers/${paperId}/method?refresh=true`, {
    method: "POST",
  });
}

export function reparsePaper(paperId: string) {
  return request<{ paper_id: string; status: string }>(`/papers/${paperId}/reparse`, {
    method: "POST",
  });
}

export function rebuildPaperIndex(paperId: string) {
  return request<Paper>(`/papers/${paperId}/index`, { method: "POST" });
}

export function searchPaper(paperId: string, q: string, k?: number) {
  const params = new URLSearchParams({ q });
  if (k) params.set("k", String(k));
  return request<Evidence[]>(`/papers/${paperId}/search?${params.toString()}`);
}

export function askPaper(paperId: string, body: AskRequest) {
  return request<PaperAnswer>(
    `/papers/${paperId}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: body.question,
        history: body.history ?? [],
      }),
    },
    180000,
  );
}

export function searchLibrary(q: string, k?: number) {
  const params = new URLSearchParams({ q });
  if (k) params.set("k", String(k));
  return request<LibraryHit[]>(`/library/search?${params.toString()}`);
}

export function askLibrary(body: AskRequest) {
  return request<PaperAnswer>(
    "/library/ask",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: body.question,
        history: body.history ?? [],
      }),
    },
    180000,
  );
}

export function figureUrl(paperId: string, figureId: string) {
  return `/api/papers/${paperId}/figures/${figureId}`;
}

export function sourcePdfUrl(paperId: string) {
  return `/api/papers/${paperId}/source`;
}

export function startTranslations(paperId: string, refresh = false) {
  const query = refresh ? "?refresh=true" : "";
  return request<PaperTranslation>(`/papers/${paperId}/translations${query}`, {
    method: "POST",
  });
}

export function getTranslations(paperId: string) {
  return request<PaperTranslation>(`/papers/${paperId}/translations`);
}

export function cancelTranslations(paperId: string) {
  return request<PaperTranslation>(`/papers/${paperId}/translations/cancel`, {
    method: "POST",
  });
}

export async function ensureTranslations(paperId: string): Promise<PaperTranslation> {
  let payload = await startTranslations(paperId);
  if (payload.status === "ready" || payload.status === "partial" || payload.status === "cancelled") {
    return payload;
  }
  if (payload.status === "failed") {
    throw new Error("翻译失败，请检查 API Key / 模型名后重试");
  }
  // pending：轮询；已有章节先返回进度由调用方多次调用也可
  for (let attempt = 0; attempt < 300; attempt += 1) {
    await new Promise((resolve) => window.setTimeout(resolve, 3000));
    payload = await getTranslations(paperId);
    if (payload.status === "ready" || payload.status === "partial" || payload.status === "cancelled") {
      return payload;
    }
    if (payload.status === "failed") {
      throw new Error("翻译失败，请检查 API Key / 模型名后重试");
    }
  }
  throw new Error("翻译超时，请稍后重试");
}

export function formatApiError(error: unknown): string {
  if (error instanceof HttpError) {
    if (error.api?.message) {
      return error.api.stage ? `${error.api.message}（${error.api.stage}）` : error.api.message;
    }
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "请求失败";
}
