import type { ChatRetrievalStatus } from "@/lib/api/chat";

import { describeRetrieval, type RetrievalTone } from "./retrievalStatus";

const TONE_CLASSES: Record<RetrievalTone, string> = {
  ok: "border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-100",
  warn: "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100",
  error: "border-red-300 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100",
};

const TONE_PREFIX: Record<RetrievalTone, string> = {
  ok: "",
  warn: "Warning: ",
  error: "Problem: ",
};

export function RetrievalBadges({ retrieval }: { retrieval: ChatRetrievalStatus }) {
  const badges = describeRetrieval(retrieval);
  if (!badges.length) {
    return null;
  }
  return (
    <ul aria-label="Answer grounding" className="mt-2 flex flex-wrap gap-1.5">
      {badges.map((badge) => (
        <li
          key={badge.label}
          title={badge.detail}
          className={`rounded-full border px-2 py-0.5 text-xs font-medium ${TONE_CLASSES[badge.tone]}`}
        >
          <span className="sr-only">{TONE_PREFIX[badge.tone]}</span>
          {badge.label}
          <span className="sr-only">. {badge.detail}</span>
        </li>
      ))}
    </ul>
  );
}
