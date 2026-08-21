import type { Figure, Health, MethodExplain, Paper, PaperIntro, Section } from "./types";
import { HttpError, type ApiError } from "./types";

export type { Figure, Health, MethodExplain, Paper, PaperIntro, Section };
export { HttpError };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init);
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

export function uploadPaper(file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<{ paper_id: string; status: string }>("/papers", {
    method: "POST",
    body,
  });
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

export function figureUrl(paperId: string, figureId: string) {
  return `/api/papers/${paperId}/figures/${figureId}`;
}

export function sourcePdfUrl(paperId: string) {
  return `/api/papers/${paperId}/source`;
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
