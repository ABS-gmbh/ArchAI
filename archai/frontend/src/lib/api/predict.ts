import { ApiError, fetchBackend, readErrorDetail } from "./client";

export interface PredictSingleResponse {
  task_id: string;
  coco_json: Record<string, unknown>;
  stats: Record<string, number>;
  annotated_image_url: string;
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
    response = await fetchBackend("/predict/single", {
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
    throw new ApiError(response.status, await readErrorDetail(response));
  }

  return (await response.json()) as PredictSingleResponse;
}
