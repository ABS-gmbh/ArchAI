import type { ChatRetrievalStatus } from "@/lib/api/chat";

export type ChatRole = "user" | "assistant" | "system";

export interface WorkspacePage {
  id: string;
  name: string;
  dataUrl: string;
  mimeType: string;
}

export interface WorkspaceDocumentMetadata {
  language: string;
  year: string;
  placeOrOrigin: string;
  scriptFamily: string;
  documentType: string;
  notes: string;
}

export interface WorkspaceDocument {
  id: string;
  name: string;
  pages: WorkspacePage[];
  createdAt: number;
  metadata: WorkspaceDocumentMetadata;
}

export interface WorkspaceChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  imageUrl?: string;
  imageAlt?: string;
  retrieval?: ChatRetrievalStatus | null;
  /** Quality badges of a finished extraction, on its status message. */
  extractionBadges?: { tone: "ok" | "warn" | "error" | "neutral"; label: string; detail: string }[];
  /** No answer came: the content says why. */
  failed?: boolean;
  /** What the verifier made of an answer, when one ran: its verdict, its own wording and its notes. */
  verification?: { assessment: string; corrected_answer: string; notes: string[] } | null;
}

export interface WorkspacePersistedState {
  selectedModel: string | null;
  includeCurrentPageImage: boolean;
}
