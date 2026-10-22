import { apiUrl } from "./client";

export interface PredictSingleResponse {
  task_id: string;
  coco_json: Record<string, unknown>;
  stats: Record<string, number>;
  annotated_image_url: string;
}

function parseErrorText(status: number, raw: string): string {
  try {
    const parsed = JSON.parse(raw) as { detail?: string };
    if (typeof parsed.detail === "string" && parsed.detail.trim()) {
      return `HTTP ${status}: ${parsed.detail}`;
    }
  } catch {
    // ignore parse errors
  }
  return `HTTP ${status}: ${raw || "Request failed"}`;
}

export async function predictSinglePage(
  image: File,
  confidence = 0.25,
  iou = 0.3,
): Promise<PredictSingleResponse> {
  const controller = new AbortController();
  const timeoutMs = 180_000;
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  const form = new FormData();
  form.append("image", image);
  form.append("confidence", String(confidence));
  form.append("iou", String(iou));

  let response: Response;
  try {
    response = await fetch(apiUrl("/predict/single"), {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
  } catch (error: unknown) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Segmentation timed out after 180 seconds. The first run can take a while while models warm up. Check the backend and retry.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const raw = await response.text();
    throw new Error(parseErrorText(response.status, raw));
  }

  return (await response.json()) as PredictSingleResponse;
}
