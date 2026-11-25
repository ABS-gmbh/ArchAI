// What the reader is shown of a chat answer: the answer itself, and what a
// second model made of it. The backend still appends "[Verification]" and
// "[Retrieval]" blocks to the answer text for clients that only read text;
// the workspace gets the same facts as data and shows them as such.

export interface AnswerVerification {
  assessment: string;
  corrected_answer: string;
  notes: string[];
  /** The model that answered also checked: the verifier's own model is not offered (2026-11-25). */
  self_check?: boolean;
}

export interface VerificationNote {
  tone: "ok" | "warn" | "error";
  label: string;
  detail: string;
  /** The checker's own wording of the answer, when it differs from the answer shown. */
  corrected: string | null;
  notes: string[];
}

const APPENDIX = /\n*\[(?:Verification|Retrieval)\]\n/;

/**
 * The answer without the appended blocks. While an answer is still streaming
 * only the cut is made; the trailing space of a half-sent sentence is kept so
 * the next piece joins on correctly.
 */
export function answerProper(text: string, streaming = false): string {
  const cut = text.search(APPENDIX);
  const answer = cut === -1 ? text : text.slice(0, cut);
  return streaming ? answer : answer.trimEnd();
}

const sameText = (a: string, b: string) => a.replace(/\s+/g, " ").trim() === b.replace(/\s+/g, " ").trim();

const VERDICTS: Record<string, Pick<VerificationNote, "tone" | "label" | "detail">> = {
  supported: {
    tone: "ok",
    label: "Checked: supported",
    detail: "A second model compared the answer with the passages shown to it and found it supported.",
  },
  partially_supported: {
    tone: "warn",
    label: "Checked: partly supported",
    detail: "A second model found only part of the answer in the passages shown to it.",
  },
  unsupported: {
    tone: "error",
    label: "Checked: not supported",
    detail: "A second model did not find the answer in the passages shown to it.",
  },
};

/**
 * The verifier's verdict in words, or null when there is none to show: no
 * verifier ran, or it could not run (the grounding badges already say so).
 */
export function describeVerification(verification: AnswerVerification | null | undefined, answer: string): VerificationNote | null {
  const verdict = verification ? VERDICTS[String(verification.assessment || "").toLowerCase()] : undefined;
  if (!verification || !verdict) {
    return null;
  }
  const corrected = String(verification.corrected_answer || "").trim();
  // The checker is given the answer with its appendix-free text; its "corrected answer" may still carry one.
  const correctedProper = answerProper(corrected);
  // A check by the model that wrote the answer is weaker than a second model's, and is called so.
  const said = verification.self_check
    ? {
        ...verdict,
        label: verdict.label.replace(/^Checked/, "Self-checked"),
        detail: `${verdict.detail.replace(/^A second model/, "The model that wrote the answer")} A model checking its own answer is weaker than a second model.`,
      }
    : verdict;
  return {
    ...said,
    corrected: correctedProper && !sameText(correctedProper, answer) ? correctedProper : null,
    notes: (verification.notes ?? []).map((note) => String(note || "").trim()).filter(Boolean).slice(0, 4),
  };
}
