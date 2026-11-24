import type { ChatRetrievalStatus } from "@/lib/api/chat";

export type RetrievalTone = "ok" | "warn" | "error";

export interface RetrievalBadge {
  tone: RetrievalTone;
  label: string;
  detail: string;
}

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/**
 * Badges for an answer's grounding, from the backend's `retrieval` object.
 *
 * Only a chat about one page (`run_scoped`) gets grounding badges: a general
 * chat has no page to be grounded in. The verifier badge is shown whenever the
 * verifier was configured but could not run.
 */
export function describeRetrieval(retrieval: ChatRetrievalStatus | null | undefined): RetrievalBadge[] {
  if (!retrieval || retrieval.status === "not_used") {
    return [];
  }
  const badges: RetrievalBadge[] = [];
  if (retrieval.run_scoped) {
    const hits = retrieval.chunk_hits ?? 0;
    const passages = plural(hits, "passage");
    // With no retrieved passage, the only page text the model saw is the
    // transcript excerpt the workspace sends along, if any.
    const fallback = retrieval.transcript_in_context
      ? " The answer can rest only on the unsearched transcript excerpt sent with the question."
      : " The answer is not grounded in this page's text.";
    switch (retrieval.status) {
      case "ok":
        badges.push(
          hits > 0
            ? {
                tone: "ok",
                label: `Page sources · ${passages}`,
                detail: "Passages retrieved from this page's transcription were given to the model as evidence.",
              }
            : {
                tone: "warn",
                label: "No matching passages",
                detail: `Search ran but found no passage of this page for the question.${fallback}`,
              },
        );
        break;
      case "degraded": {
        const label =
          retrieval.chunk_mode === "lexical"
            ? "Keyword search only"
            : retrieval.chunk_mode === "dense"
              ? "Semantic search only"
              : "Search partly unavailable";
        badges.push({
          tone: "warn",
          label: `${label} · ${passages}`,
          detail: `Part of the search did not run, so relevant evidence may have been missed.${hits > 0 ? "" : fallback}`,
        });
        break;
      }
      case "no_searchable_run":
        badges.push({
          tone: "warn",
          label: "No searchable transcription",
          detail: `No transcription of this page has passed the quality gate, so it was not searched.${fallback}`,
        });
        break;
      case "unavailable":
        badges.push({
          tone: "error",
          label: "Search unavailable",
          detail: `The page's sources could not be searched.${fallback}`,
        });
        break;
    }
  }
  if (retrieval.verifier === "unavailable") {
    badges.push({
      tone: "warn",
      label: "Not verified",
      detail: "The answer verifier was unavailable, so citations were not checked.",
    });
  } else if (retrieval.verifier === "ran" && retrieval.run_scoped) {
    // "Checked", not "Verified": the verifier ran, which says nothing about its verdict.
    badges.push({
      tone: "ok",
      label: "Checked by verifier",
      detail: "A second model checked this answer against the evidence; its findings, if any, follow the answer.",
    });
  }
  return badges;
}
