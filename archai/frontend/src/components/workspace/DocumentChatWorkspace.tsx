"use client";

import { type ChangeEvent, type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  analyzeSegmentLabel,
  createChatCompletion,
  getChatModels,
  type ChatMessagePayload,
} from "@/lib/api/chat";
import { ApiError } from "@/lib/api/client";
import {
  type FullPageLine,
  type RunSnapshot,
  extractPageText,
  fetchAuthorityReport,
  fetchLinkingReport,
  followRun,
  mergeOCRResultText,
  reserveRun,
} from "@/lib/api/ocrAgent";
import { predictSinglePage } from "@/lib/api/predict";
import {
  type WorkspaceChatMessage,
  type WorkspaceDocument,
  type WorkspaceDocumentMetadata,
  type WorkspacePage,
  type WorkspacePersistedState,
} from "@/lib/workspace/types";
import {
  clamp,
  makeId,
  pageToFile,
  pageTooLargeMessage,
  readFileAsDataUrl,
  sortedImages,
  toBase64,
} from "./workspaceFiles";
import {
  type CocoPayload,
  type StoredLabelRegionMap,
  availableStoredLabels,
  bestMatchingSegmentationLabel,
  buildSegmentationSummary,
  buildStoredLabelRegions,
  buildTransparentCropOverlay,
  resolveCropLabelFromPrompt,
} from "./workspaceSegmentation";
import {
  buildContextualUserPrompt,
  buildEnglishTranslationPrompt,
  detectWorkspaceIntent,
  isLabelAnalysisPrompt,
  resolveOcrPromptHints,
} from "./workspaceIntents";
import { BackendStatusBanner } from "./BackendStatusBanner";
import { ChatPanel } from "./ChatPanel";
import { DocumentPanel } from "./DocumentPanel";
import { EntitiesPanel, type LinkingState } from "./EntitiesPanel";
import { MetadataDialog } from "./MetadataDialog";
import { PageViewer } from "./PageViewer";
import { RunPanel, stageInWords } from "./RunPanel";
import { IDLE_READ_STATE, type ReadState, type TranscriptionLine, TranscriptionPane } from "./TranscriptionPane";
import { Button, Icon, Mark } from "./ui";
import {
  type ExtractionBadge,
  describeExtraction,
  getExtractionStatus,
  cleanTranscription,
} from "./workspaceExtraction";

interface DocumentChatWorkspaceProps {
  initialDocumentId?: string;
}

const STORAGE_KEY = "archai_workspace_state_v1";
const DEFAULT_CHAT_MODEL_ID = "qwen3-30b-a3b-instruct-2507";
const DEFAULT_TASK_MODELS = {
  ocr_model: "glm-ocr:latest",
  chat_rag_model: DEFAULT_CHAT_MODEL_ID,
  translation_model: "llama-3.3-70b-instruct",
  label_visual_model: "qwen3-vl-30b-a3b-instruct",
  label_visual_fallback_model: "internvl3.5-30b-a3b",
  verifier_model: "qwen3-235b-a22b",
  embedding_model: "multilingual-e5-large-instruct",
};
const SEGMENTATION_LABEL_VOCAB = [
  "Border",
  "Table",
  "Diagram",
  "Main script black",
  "Main script coloured",
  "Variant script black",
  "Variant script coloured",
  "Historiated",
  "Inhabited",
  "Zoo - Anthropomorphic",
  "Embellished",
  "Plain initial- coloured",
  "Plain initial - Highlighted",
  "Plain initial - Black",
  "Page Number",
  "Quire Mark",
  "Running header",
  "Catchword",
  "Gloss",
  "Illustrations",
  "Column",
  "GraphicZone",
  "MusicLine",
  "MusicZone",
  "Music",
];

type WorkspacePane = "page" | "text" | "ask" | "names" | "run" | "about";

interface PendingUploadDocument {
  baseName: string;
  pages: WorkspacePage[];
}
const EMPTY_DOCUMENT_METADATA: WorkspaceDocumentMetadata = {
  language: "",
  year: "",
  placeOrOrigin: "",
  scriptFamily: "",
  documentType: "",
  notes: "",
};

/**
 * Focus the first element matching *selector* once it is on screen, unless
 * focus has gone somewhere meaningful meanwhile (anywhere but the body). For a
 * control that disappears or is disabled while it has focus.
 */
function focusWhenShown(selector: string, frames = 20): void {
  requestAnimationFrame(() => {
    if (document.activeElement && document.activeElement !== document.body) {
      return;
    }
    const target = document.querySelector<HTMLElement>(selector);
    if (target) {
      target.focus();
    } else if (frames > 0) {
      focusWhenShown(selector, frames - 1);
    }
  });
}

export function DocumentChatWorkspace({ initialDocumentId }: DocumentChatWorkspaceProps) {
  const [clientReady, setClientReady] = useState(false);
  const [hydrated, setHydrated] = useState(false);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
  const [pageIndexByDocument, setPageIndexByDocument] = useState<Record<string, number>>({});
  const [zoomByDocument, setZoomByDocument] = useState<Record<string, number>>({});
  const [messagesByDocument, setMessagesByDocument] = useState<Record<string, WorkspaceChatMessage[]>>({});

  const [visionModelIds, setVisionModelIds] = useState<Set<string>>(new Set());
  const [taskModels, setTaskModels] = useState(DEFAULT_TASK_MODELS);
  const [selectedModel, setSelectedModel] = useState<string>(DEFAULT_CHAT_MODEL_ID);
  const [includeCurrentPageImage, setIncludeCurrentPageImage] = useState(false);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [assistantLoadingLabel, setAssistantLoadingLabel] = useState("Thinking");
  const [error, setError] = useState<string | null>(null);
  const [segmentedPreviewByPageId, setSegmentedPreviewByPageId] = useState<Record<string, string>>({});
  const [segmentationLabelsByPageId, setSegmentationLabelsByPageId] = useState<Record<string, StoredLabelRegionMap>>({});
  const [segmentationErrorByPageId, setSegmentationErrorByPageId] = useState<Record<string, string>>({});
  const [ocrTextByPageId, setOcrTextByPageId] = useState<Record<string, string>>({});
  const [ocrRunIdByPageId, setOcrRunIdByPageId] = useState<Record<string, string>>({});
  const [authorityReportByPageId, setAuthorityReportByPageId] = useState<Record<string, string>>({});
  const [segmentingPageId, setSegmentingPageId] = useState<string | null>(null);
  // What the reading of each page produced, for the viewer and the transcription pane.
  const [pageLinesByPageId, setPageLinesByPageId] = useState<Record<string, FullPageLine[]>>({});
  const [textLinesByPageId, setTextLinesByPageId] = useState<Record<string, TranscriptionLine[]>>({});
  const [marginLinesByPageId, setMarginLinesByPageId] = useState<Record<string, string[]>>({});
  const [pageSizeByPageId, setPageSizeByPageId] = useState<Record<string, { width: number; height: number }>>({});
  const [readStateByPageId, setReadStateByPageId] = useState<Record<string, ReadState>>({});
  const [badgesByPageId, setBadgesByPageId] = useState<Record<string, ExtractionBadge[]>>({});
  const [runSnapshotByPageId, setRunSnapshotByPageId] = useState<Record<string, RunSnapshot>>({});
  const [linkingByRunId, setLinkingByRunId] = useState<Record<string, LinkingState>>({});
  const [openError, setOpenError] = useState<string | null>(null);
  const [activeLineId, setActiveLineId] = useState<string | null>(null);
  const [scrollToLineId, setScrollToLineId] = useState<string | null>(null);
  // Which region is shown where the screen has room for one: the page, its text, or a panel.
  const [pane, setPane] = useState<WorkspacePane>("text");
  const runFollowersRef = useRef<Set<AbortController>>(new Set());
  // The run each page's latest read reserved: an older run's follower may not write over it.
  const latestRunByPageRef = useRef<Record<string, string | null>>({});
  // The control that opened the file picker, to give focus back to when the dialog closes.
  const dialogOpenerRef = useRef<HTMLElement | null>(null);
  const [showSegmentationOverlay, setShowSegmentationOverlay] = useState(true);
  const [pendingUpload, setPendingUpload] = useState<PendingUploadDocument | null>(null);
  const [metadataDraft, setMetadataDraft] = useState<WorkspaceDocumentMetadata>(EMPTY_DOCUMENT_METADATA);
  const [metadataError, setMetadataError] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const [replyAnnouncement, setReplyAnnouncement] = useState("");
  const wasSendingRef = useRef(false);
  /** Whether the last chat request failed; its message is in the conversation, not above the composer. */
  const chatFailedRef = useRef(false);
  const metadataDialogWasOpenRef = useRef(false);

  const currentDocument = useMemo(
    () => documents.find((doc) => doc.id === selectedDocumentId) ?? null,
    [documents, selectedDocumentId],
  );

  const currentPageIndex = currentDocument
    ? clamp(pageIndexByDocument[currentDocument.id] ?? 0, 0, Math.max(0, currentDocument.pages.length - 1))
    : 0;

  const currentPage = currentDocument ? currentDocument.pages[currentPageIndex] ?? null : null;
  const currentZoom = currentDocument ? zoomByDocument[currentDocument.id] ?? 1 : 1;
  const currentMessages = currentDocument ? (messagesByDocument[currentDocument.id] ?? []) : [];
  const currentSegmentedPreview = currentPage ? (segmentedPreviewByPageId[currentPage.id] ?? null) : null;
  const currentSegmentationLabels = currentPage ? (segmentationLabelsByPageId[currentPage.id] ?? null) : null;
  const currentSegmentationError = currentPage ? (segmentationErrorByPageId[currentPage.id] ?? null) : null;
  const currentExtractedText = currentPage ? (ocrTextByPageId[currentPage.id] ?? "") : "";
  const currentOcrRunId = currentPage ? (ocrRunIdByPageId[currentPage.id] ?? "") : "";
  const currentAuthorityReport = currentPage ? (authorityReportByPageId[currentPage.id] ?? "") : "";
  const currentMetadata = currentDocument?.metadata ?? null;

  useEffect(() => {
    setClientReady(true);
  }, []);

  useEffect(() => {
    if (sending) {
      // A page read or a region search also sets `sending`; only a chat request ends in a reply.
      wasSendingRef.current = assistantLoadingLabel !== "Extracting text" && assistantLoadingLabel !== "Segmenting page";
      setReplyAnnouncement("");
    } else if (wasSendingRef.current) {
      wasSendingRef.current = false;
      setReplyAnnouncement(error || chatFailedRef.current ? "The request failed." : "Reply ready.");
    }
  }, [sending, error, assistantLoadingLabel]);

  // When the metadata dialog closes, hand focus back to the control that
  // opened it; otherwise it falls to the page body and a keyboard user starts
  // again from the top.
  useEffect(() => {
    if (pendingUpload) {
      metadataDialogWasOpenRef.current = true;
    } else if (metadataDialogWasOpenRef.current) {
      metadataDialogWasOpenRef.current = false;
      const opener = dialogOpenerRef.current;
      dialogOpenerRef.current = null;
      if (opener && opener.isConnected) {
        opener.focus();
      } else {
        // The opener is gone (the first-run button, once a document is open).
        document.querySelector<HTMLElement>("[data-open-pages]")?.focus();
      }
    }
  }, [pendingUpload]);

  useEffect(() => {
    let cancelled = false;
    getChatModels()
      .then((payload) => {
        if (cancelled) {
          return;
        }
        setVisionModelIds(new Set(payload.vision_models));
        setTaskModels(payload.task_models || DEFAULT_TASK_MODELS);
        setSelectedModel(payload.default_model || payload.task_models?.chat_rag_model || DEFAULT_CHAT_MODEL_ID);
      })
      .catch(() => {
        // The defaults stay in place. Why the backend did not answer is the
        // status banner's to say, not the chat's.
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) {
        setHydrated(true);
        return;
      }

      const parsed = JSON.parse(raw) as WorkspacePersistedState;
      setIncludeCurrentPageImage(Boolean(parsed.includeCurrentPageImage));
      setSelectedModel(DEFAULT_CHAT_MODEL_ID);
    } catch {
      // Ignore malformed persisted state.
    } finally {
      setHydrated(true);
    }
  }, [initialDocumentId]);

  useEffect(() => {
    if (!hydrated) {
      return;
    }
    const payload: WorkspacePersistedState = {
      selectedModel: selectedModel || null,
      includeCurrentPageImage,
    };
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // Storage quota/availability should never block workspace usage.
    }
  }, [
    hydrated,
    selectedModel,
    includeCurrentPageImage,
  ]);

  useEffect(() => {
    if (!initialDocumentId || !documents.some((doc) => doc.id === initialDocumentId)) {
      return;
    }
    setSelectedDocumentId(initialDocumentId);
  }, [initialDocumentId, documents]);

  const selectDocument = (documentId: string) => {
    setSelectedDocumentId(documentId);
  };

  // Pages arrive from the file picker or by being dropped on the viewer.
  const openFiles = async (files: FileList | File[] | null) => {
    if (!files || files.length === 0) {
      return;
    }

    try {
      setOpenError(null);
      // Browsers cannot draw TIFF, so a TIFF page would be a broken image with its line boxes adrift.
      const all = sortedImages(files);
      const images = all.filter((file) => !/tiff?$/i.test(file.type) && !/\.tiff?$/i.test(file.name));
      const skipped = all.length - images.length;
      if (!images.length) {
        setOpenError(
          skipped
            ? "TIFF images cannot be shown in a browser. Convert the pages to PNG or JPEG and open those."
            : "None of those files is an image. Choose PNG or JPEG page images.",
        );
        return;
      }
      if (skipped) {
        setOpenError(`${skipped} TIFF ${skipped === 1 ? "image was" : "images were"} left out: a browser cannot show TIFF. Convert ${skipped === 1 ? "it" : "them"} to PNG or JPEG.`);
      }

      const pages: WorkspacePage[] = await Promise.all(
        images.map(async (file, index) => ({
          id: makeId(`page-${index + 1}`),
          name: file.name,
          dataUrl: await readFileAsDataUrl(file),
          mimeType: file.type || "image/png",
        })),
      );

      const baseName = images[0].name.replace(/\.[^.]+$/, "") || "Document";
      setPendingUpload({ baseName, pages });
      setMetadataDraft({
        ...EMPTY_DOCUMENT_METADATA,
        scriptFamily: "medieval latin script",
        documentType: "manuscript",
      });
      setMetadataError(null);
    } catch (err) {
      setOpenError(err instanceof Error ? `The files could not be opened. ${err.message}` : "The files could not be opened.");
    }
  };

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    try {
      await openFiles(event.target.files);
    } finally {
      event.target.value = "";
    }
  };

  const commitPendingUpload = () => {
    if (!pendingUpload) {
      return;
    }

    const language = metadataDraft.language.trim();
    const year = metadataDraft.year.trim();
    if (!language || !year) {
      setMetadataError("Language and year are required before extraction can begin.");
      return;
    }

    const document: WorkspaceDocument = {
      id: makeId("doc"),
      name: pendingUpload.pages.length > 1 ? `${pendingUpload.baseName} (${pendingUpload.pages.length} pages)` : pendingUpload.baseName,
      pages: pendingUpload.pages,
      createdAt: Date.now(),
      metadata: {
        language,
        year,
        placeOrOrigin: metadataDraft.placeOrOrigin.trim(),
        scriptFamily: metadataDraft.scriptFamily.trim(),
        documentType: metadataDraft.documentType.trim(),
        notes: metadataDraft.notes.trim(),
      },
    };

    setDocuments((prev) => [document, ...prev]);
    setPageIndexByDocument((prev) => ({ ...prev, [document.id]: 0 }));
    setZoomByDocument((prev) => ({ ...prev, [document.id]: 1 }));
    setMessagesByDocument((prev) => ({ ...prev, [document.id]: [] }));
    setSelectedDocumentId(document.id);
    setPendingUpload(null);
    setMetadataDraft(EMPTY_DOCUMENT_METADATA);
    setMetadataError(null);
  };

  const cancelPendingUpload = () => {
    setPendingUpload(null);
    setMetadataDraft(EMPTY_DOCUMENT_METADATA);
    setMetadataError(null);
  };

  const updateCurrentPageIndex = (next: number) => {
    if (!currentDocument) {
      return;
    }
    setPageIndexByDocument((prev) => ({
      ...prev,
      [currentDocument.id]: clamp(next, 0, Math.max(0, currentDocument.pages.length - 1)),
    }));
  };

  const updateZoom = (next: number) => {
    if (!currentDocument) {
      return;
    }
    setZoomByDocument((prev) => ({
      ...prev,
      [currentDocument.id]: clamp(next, 0.5, 2.5),
    }));
  };

  const clearConversation = () => {
    if (!currentDocument) {
      return;
    }
    setMessagesByDocument((prev) => ({ ...prev, [currentDocument.id]: [] }));
  };

  const appendMessages = (documentId: string, nextMessages: WorkspaceChatMessage[]) => {
    setMessagesByDocument((prev) => ({
      ...prev,
      [documentId]: [...(prev[documentId] ?? []), ...nextMessages],
    }));
  };

  const sendPromptToChat = async (
    text: string,
    options?: {
      displayText?: string;
      attachImage?: boolean;
      forceAttachImage?: boolean;
      forcedImageDataUrl?: string;
      loadingLabel?: string;
      modelOverride?: string;
      chatStage?: string;
      historyForModel?: WorkspaceChatMessage[];
      /** What a read that just finished produced; the render's own values are still the old ones. */
      pageContext?: { transcript: string; runId: string; authorityReport: string };
      /** The question is already in the conversation (typed in the composer): do not add it again. */
      questionShown?: boolean;
    },
  ): Promise<{ ok: boolean; error?: string }> => {
    if (!currentDocument || sending) {
      return { ok: false, error: "No active document or request already in progress." };
    }

    if (!text) {
      return { ok: false, error: "Prompt is empty." };
    }

    const explicitImageAttach = options?.attachImage;
    const forcedImageAttach = Boolean(options?.forceAttachImage);
    const shouldAttachImage =
      explicitImageAttach !== undefined
        ? explicitImageAttach
        : (forcedImageAttach || includeCurrentPageImage);
    const imageDataUrl = options?.forcedImageDataUrl ?? currentPage?.dataUrl ?? null;
    const requestedModel = (options?.modelOverride || selectedModel || taskModels.chat_rag_model || DEFAULT_CHAT_MODEL_ID).trim();
    let modelForRequest = requestedModel;
    const priorMessagesForModel = [...(options?.historyForModel ?? currentMessages)];
    const chatStage = options?.chatStage || (shouldAttachImage ? "visual_chat" : "rag_chat");

    if (shouldAttachImage && !imageDataUrl) {
      setError("No current page image available to attach.");
      return { ok: false, error: "No current page image available to attach." };
    }
    if (shouldAttachImage) {
      if (!modelForRequest || !visionModelIds.has(modelForRequest)) {
        const fallbackVisionModel =
          [taskModels.label_visual_model, taskModels.label_visual_fallback_model]
            .find((candidate) => candidate && visionModelIds.has(candidate))
          || "";
        if (!fallbackVisionModel) {
          setError("No vision-capable chat model is available on the backend.");
          return { ok: false, error: "No vision-capable chat model is available." };
        }
        modelForRequest = fallbackVisionModel;
      }
    }
    if (!modelForRequest) {
      setError("No chat model selected.");
      return { ok: false, error: "No chat model selected." };
    }

    const displayText = options?.displayText ?? text;
    const userMessage: WorkspaceChatMessage = {
      id: makeId("msg-user"),
      role: "user",
      content: displayText,
      createdAt: Date.now(),
    };
    const assistantMessage: WorkspaceChatMessage = {
      id: makeId("msg-assistant"),
      role: "assistant",
      content: "",
      createdAt: Date.now(),
    };

    setMessagesByDocument((prev) => ({
      ...prev,
      [currentDocument.id]: [
        ...(prev[currentDocument.id] ?? []),
        ...(options?.questionShown ? [] : [userMessage]),
        assistantMessage,
      ],
    }));
    setError(null);
    chatFailedRef.current = false;
    setAssistantLoadingLabel(options?.loadingLabel || "Thinking");
    setSending(true);

    const apiMessages: ChatMessagePayload[] = [
      ...priorMessagesForModel.map((message) => ({
        role: message.role,
        content: message.content,
      })),
      shouldAttachImage && imageDataUrl
        ? {
            role: "user",
            content: [
              { type: "text", text },
              { type: "image_url", image_url: { url: imageDataUrl } },
            ],
          }
        : {
            role: "user",
            content: text,
          },
    ];

    try {
      const result = await createChatCompletion(
        {
          messages: apiMessages,
          model: modelForRequest,
          temperature: 0.2,
          stream: true,
          context: {
            chat_stage: chatStage,
            document_id: currentDocument.id,
            filename: currentPage?.name || currentDocument.name,
            current_page_index: currentPageIndex,
            page_count: currentDocument.pages.length,
            transcript: (options?.pageContext?.transcript ?? currentExtractedText) || undefined,
            authority_report:
              chatStage === "translation"
                ? undefined
                : (options?.pageContext?.authorityReport ?? currentAuthorityReport) || undefined,
            ocr_run_id: (options?.pageContext?.runId ?? currentOcrRunId) || undefined,
            document_language: currentMetadata?.language || undefined,
            document_year: currentMetadata?.year || undefined,
            place_or_origin: currentMetadata?.placeOrOrigin || undefined,
            script_family: currentMetadata?.scriptFamily || undefined,
            document_type: currentMetadata?.documentType || undefined,
            document_notes: currentMetadata?.notes || undefined,
          },
        },
        (delta) => {
          setMessagesByDocument((prev) => {
            const list = [...(prev[currentDocument.id] ?? [])];
            const index = list.findIndex((msg) => msg.id === assistantMessage.id);
            if (index === -1) {
              return prev;
            }
            list[index] = { ...list[index], content: `${list[index].content}${delta}` };
            return { ...prev, [currentDocument.id]: list };
          });
        },
      );

      if (result.text) {
        setMessagesByDocument((prev) => {
          const list = [...(prev[currentDocument.id] ?? [])];
          const index = list.findIndex((msg) => msg.id === assistantMessage.id);
          if (index === -1) {
            return prev;
          }
          list[index] = {
            ...list[index],
            content: result.incomplete
              ? `${result.text}\n\n[Incomplete answer: the connection closed before the answer finished. It was not verified.]`
              : result.text,
            retrieval: result.retrieval ?? null,
          };
          return { ...prev, [currentDocument.id]: list };
        });
      }
      if (result.incomplete) {
        setError("The answer was cut off before it finished. Ask again to get the full answer.");
        if (!result.text) {
          throw new Error("The connection closed before any answer arrived.");
        }
      }
      return { ok: true };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Chat request failed.";
      // The failure is shown once, where the answer would have been.
      chatFailedRef.current = true;
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocument.id] ?? [])];
        const index = list.findIndex((msg) => msg.id === assistantMessage.id);
        if (index !== -1) {
          list[index] = {
            ...list[index],
            content: `No answer: ${message}`,
            failed: true,
          };
        }
        return { ...prev, [currentDocument.id]: list };
      });
      return { ok: false, error: message };
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const sendMessage = async () => {
    const text = input.trim();
    const documentId = currentDocument?.id ?? null;
    if (!text || !documentId) {
      return;
    }
    setInput("");
    const userMessage: WorkspaceChatMessage = {
      id: makeId("msg-user"),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };

    const intent = detectWorkspaceIntent(text);
    const matchedLabel =
      resolveCropLabelFromPrompt(text, currentSegmentationLabels)
      || bestMatchingSegmentationLabel(text, SEGMENTATION_LABEL_VOCAB);

    if (!intent && matchedLabel && isLabelAnalysisPrompt(text)) {
      appendMessages(documentId, [userMessage]);
      await handleLabelAnalysisInChat(text);
      return;
    }

    if (intent === "segment") {
      appendMessages(documentId, [userMessage]);
      await handleSegmentationInChat();
      return;
    }

    if (intent === "crop") {
      appendMessages(documentId, [userMessage]);
      await handleCropInChat(text);
      return;
    }

    if (intent === "extract") {
      appendMessages(documentId, [userMessage]);
      await handleExtractTextInChat({ userPrompt: text });
      return;
    }

    if (intent === "translate") {
      appendMessages(documentId, [userMessage]);
      await handleTranslateInChat(text, true);
      return;
    }

    if (intent === "entities") {
      appendMessages(documentId, [userMessage]);
      await handleEntityQuestionInChat(text, true);
      return;
    }

    await sendPromptToChat(text, { loadingLabel: "Thinking" });
  };

  const runSegmentationForCurrentPage = async (): Promise<{ previewUrl: string; coco: CocoPayload; labelsByName: StoredLabelRegionMap } | null> => {
    if (!currentPage) {
      return null;
    }
    setError(null);
    setSegmentingPageId(currentPage.id);
    setSegmentationErrorByPageId((prev) => {
      const next = { ...prev };
      delete next[currentPage.id];
      return next;
    });

    try {
      const pageFile = await pageToFile(currentPage);
      const result = await predictSinglePage(pageFile);
      const suffix = result.annotated_image_url.includes("?") ? "&" : "?";
      const url = `${result.annotated_image_url}${suffix}t=${Date.now()}`;
      setSegmentedPreviewByPageId((prev) => ({ ...prev, [currentPage.id]: url }));
      const coco = (result.coco_json || {}) as CocoPayload;
      const labelsByName = buildStoredLabelRegions(coco);
      setSegmentationLabelsByPageId((prev) => ({ ...prev, [currentPage.id]: labelsByName }));
      return { previewUrl: url, coco, labelsByName };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Segmentation failed.";
      setSegmentationErrorByPageId((prev) => ({ ...prev, [currentPage.id]: message }));
      setError(message);
      return null;
    } finally {
      setSegmentingPageId((prev) => (prev === currentPage.id ? null : prev));
    }
  };

  const handleLabelAnalysisInChat = async (userText: string) => {
    if (!currentPage || !currentDocument) {
      return;
    }

    const currentDocumentId = currentDocument.id;
    const statusMessageId = makeId("msg-status-label-analysis");
    appendMessages(currentDocumentId, [
      {
        id: statusMessageId,
        role: "assistant",
        content: "Label analysis status: preparing cropped label image...",
        createdAt: Date.now(),
      },
    ]);

    const setLabelStatus = (content: string) => {
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocumentId] ?? [])];
        const index = list.findIndex((msg) => msg.id === statusMessageId);
        if (index === -1) {
          return prev;
        }
        list[index] = { ...list[index], content };
        return { ...prev, [currentDocumentId]: list };
      });
    };

    setError(null);
    setSending(true);
    setAssistantLoadingLabel("Analyzing label");

    try {
      let labelsByName = currentSegmentationLabels;
      if (!labelsByName) {
        setLabelStatus("Label analysis status: running segmentation first...");
        const segmentation = await runSegmentationForCurrentPage();
        labelsByName = segmentation?.labelsByName ?? null;
      }

      if (!labelsByName) {
        setLabelStatus("Label analysis failed: segmentation data is unavailable.");
        return;
      }

      const label = resolveCropLabelFromPrompt(userText, labelsByName);
      if (!label) {
        const labels = availableStoredLabels(labelsByName);
        appendMessages(currentDocumentId, [
          {
            id: makeId("msg-assistant-label-analysis-no-match"),
            role: "assistant",
            content: labels.length
              ? `Label analysis failed: no label matched your request. Available labels: ${labels.join(", ")}`
              : "Label analysis failed: no labels are available on this page.",
            createdAt: Date.now(),
          },
        ]);
        setLabelStatus("Label analysis failed: no matching label found.");
        return;
      }

      const regions = labelsByName[label] || [];
      if (!regions.length) {
        setLabelStatus(`Label analysis failed: no stored coordinates found for "${label}".`);
        return;
      }

      setLabelStatus(`Label analysis status: cropping "${label}" and analyzing it...`);
      const response = await analyzeSegmentLabel({
        question: userText,
        label_name: label,
        image_b64: toBase64(currentPage.dataUrl),
        regions,
        filename: currentPage.name,
        page_id: currentPage.id,
        document_id: currentDocumentId,
      });
      appendMessages(currentDocumentId, [
        {
          id: makeId("msg-assistant-label-analysis"),
          role: "assistant",
          content: response.text,
          createdAt: Date.now(),
          imageUrl: `data:image/png;base64,${response.crop_image_b64}`,
          imageAlt: `${label} crop`,
        },
      ]);
      if (response.warnings.length) {
        setError(response.warnings.join(" | "));
      }
      setLabelStatus(
        response.analysis_mode
          ? `Label analysis complete for "${label}" in ${response.analysis_mode} mode using ${response.model_used}.`
          : `Label analysis complete for "${label}" using ${response.model_used}.`,
      );
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Label analysis request failed.";
      setError(message);
      setLabelStatus(`Label analysis failed: ${message}`);
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const handleSegmentationInChat = async () => {
    if (!currentPage || !currentDocument) {
      return;
    }
    const currentDocumentId = currentDocument.id;
    const statusMessageId = makeId("msg-status-seg");
    appendMessages(currentDocumentId, [
      {
        id: statusMessageId,
        role: "assistant",
        content: "Segmentation status: running page segmentation...",
        createdAt: Date.now(),
      },
    ]);

    const setSegmentationStatus = (content: string) => {
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocumentId] ?? [])];
        const index = list.findIndex((msg) => msg.id === statusMessageId);
        if (index === -1) {
          return prev;
        }
        list[index] = { ...list[index], content };
        return { ...prev, [currentDocumentId]: list };
      });
    };

    setSending(true);
    setAssistantLoadingLabel("Segmenting page");
    try {
      const segmentation = await runSegmentationForCurrentPage();
      if (!segmentation) {
        setSegmentationStatus("Segmentation failed.");
        return;
      }
      appendMessages(currentDocumentId, [
        {
          id: makeId("msg-assistant-segmentation"),
          role: "assistant",
          content: buildSegmentationSummary(segmentation.coco),
          createdAt: Date.now(),
          imageUrl: segmentation.previewUrl,
          imageAlt: currentPage.name,
        },
      ]);
      setSegmentationStatus("Segmentation complete. Summary posted to chat.");
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const handleCropInChat = async (userText: string) => {
    if (!currentPage || !currentDocument) {
      return;
    }

    const currentDocumentId = currentDocument.id;
    const statusMessageId = makeId("msg-status-crop");
    appendMessages(currentDocumentId, [
      {
        id: statusMessageId,
        role: "assistant",
        content: "Crop status: resolving segmentation labels...",
        createdAt: Date.now(),
      },
    ]);

    const setCropStatus = (content: string) => {
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocumentId] ?? [])];
        const index = list.findIndex((msg) => msg.id === statusMessageId);
        if (index === -1) {
          return prev;
        }
        list[index] = { ...list[index], content };
        return { ...prev, [currentDocumentId]: list };
      });
    };

    setSending(true);
    setAssistantLoadingLabel("Cropping label");
    setError(null);

    try {
      let labelsByName = currentSegmentationLabels;
      if (!labelsByName) {
        setCropStatus("Crop status: running segmentation first...");
        const segmentation = await runSegmentationForCurrentPage();
        labelsByName = segmentation?.labelsByName ?? null;
      }

      if (!labelsByName) {
        setCropStatus("Crop failed: segmentation data is unavailable.");
        return;
      }

      const label = resolveCropLabelFromPrompt(userText, labelsByName);
      if (!label) {
        const labels = availableStoredLabels(labelsByName);
        appendMessages(currentDocumentId, [
          {
            id: makeId("msg-assistant-crop-no-match"),
            role: "assistant",
            content: labels.length
              ? `Crop failed: no label matched your request. Available labels: ${labels.join(", ")}`
              : "Crop failed: no labels are available on this page.",
            createdAt: Date.now(),
          },
        ]);
        setCropStatus("Crop failed: no matching label found.");
        return;
      }

      setCropStatus(`Crop status: isolating "${label}" from the page...`);
      const cropped = await buildTransparentCropOverlay(currentPage.dataUrl, labelsByName, label);
      appendMessages(currentDocumentId, [
        {
          id: makeId("msg-assistant-crop-image"),
          role: "assistant",
          content: `Cropped ${cropped.matchCount} region${cropped.matchCount === 1 ? "" : "s"} for label "${label}".`,
          createdAt: Date.now(),
          imageUrl: cropped.imageUrl,
          imageAlt: `Crop for ${label}`,
        },
      ]);
      setCropStatus(`Crop complete for "${label}".`);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Crop request failed.";
      setError(message);
      setCropStatus(`Crop failed: ${message}`);
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const handleExtractTextInChat = async (options?: {
    includeDebugOutput?: boolean;
    silent?: boolean;
    userPrompt?: string;
  }): Promise<{ text: string; runId: string; authorityReport: string } | null> => {
    if (!currentPage || !currentDocument) {
      return null;
    }
    const silent = options?.silent ?? false;
    const userPrompt = options?.userPrompt ?? "";
    const currentDocumentId = currentDocument.id;

    const statusMessageId = makeId("msg-status");
    const assistantMessageId = makeId("msg-assistant-extract");
    const statusMessage: WorkspaceChatMessage = {
      id: statusMessageId,
      role: "assistant",
      content: "Extraction status: preparing OCR request...",
      createdAt: Date.now(),
    };
    const assistantMessage: WorkspaceChatMessage = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      createdAt: Date.now(),
    };
    if (!silent) {
      setMessagesByDocument((prev) => ({
        ...prev,
        [currentDocumentId]: [...(prev[currentDocumentId] ?? []), statusMessage, assistantMessage],
      }));
    }

    const setExtractionStatus = (content: string, extractionBadges?: ExtractionBadge[]) => {
      if (silent) {
        return;
      }
      setMessagesByDocument((prev) => {
        const list = [...(prev[currentDocumentId] ?? [])];
        const index = list.findIndex((msg) => msg.id === statusMessageId);
        if (index === -1) {
          return prev;
        }
        list[index] = { ...list[index], content, extractionBadges };
        return { ...prev, [currentDocumentId]: list };
      });
    };

    const pageId = currentPage.id;
    const patchRead = (patch: Partial<ReadState>) =>
      setReadStateByPageId((prev) => ({ ...prev, [pageId]: { ...(prev[pageId] ?? IDLE_READ_STATE), ...patch } }));
    patchRead({ phase: "reading", progress: null, stage: "Preparing the page", error: null });
    let follower: AbortController | null = null;

    setSending(true);
    setAssistantLoadingLabel("Extracting text");
    try {
      setExtractionStatus("Extraction status: transcribing the page...");

      // A reserved run lets this client follow its own request: lines read so
      // far, then each later step. Without one (an older backend) the page is
      // read all the same, only without live progress.
      let reservedRunId: string | null = null;
      try {
        reservedRunId = await reserveRun();
      } catch {
        reservedRunId = null;
      }
      latestRunByPageRef.current[pageId] = reservedRunId;
      if (reservedRunId) {
        const runId = reservedRunId;
        const controller = new AbortController();
        follower = controller;
        runFollowersRef.current.add(controller);
        void followRun(
          runId,
          (snapshot) => {
            if (latestRunByPageRef.current[pageId] !== runId) {
              return;
            }
            setRunSnapshotByPageId((prev) => ({ ...prev, [pageId]: snapshot }));
            const events = snapshot.events ?? [];
            const stage = snapshot.run?.current_stage ?? events[events.length - 1]?.stage ?? null;
            let progress: ReadState["progress"] = null;
            for (const event of events) {
              const match = /^lines (\d+)\/(\d+)$/.exec(event.message);
              if (match) {
                progress = { done: Number(match[1]), total: Number(match[2]) };
              }
            }
            setReadStateByPageId((prev) => {
              const current = prev[pageId] ?? IDLE_READ_STATE;
              if (current.phase !== "reading") {
                return prev;
              }
              const readingLines = String(stage || "").toUpperCase() === "OCR_RUNNING";
              return {
                ...prev,
                [pageId]: {
                  ...current,
                  // Until the first count arrives the page's layout is being found, which can take a minute on a CPU.
                  stage: readingLines && !progress ? "Finding the lines on the page" : stageInWords(stage),
                  progress: readingLines ? progress : null,
                },
              };
            });
          },
          controller.signal,
        ).finally(() => runFollowersRef.current.delete(controller));
      }

      const ocrHints = resolveOcrPromptHints(userPrompt, currentMetadata);
      const requestPage = (runId: string | null) =>
        extractPageText({
          run_id: runId ?? undefined,
          document_id: currentDocumentId,
          image_id: currentDocumentId,
          page_id: pageId,
          image_b64: toBase64(currentPage.dataUrl),
          script_hint_seed: ocrHints.scriptHintSeed,
          language_hint: ocrHints.languageHint,
          ocr_backend: "auto",
          apply_proofread: false,
          metadata: currentMetadata
            ? {
                language: currentMetadata.language,
                year: currentMetadata.year,
                place_or_origin: currentMetadata.placeOrOrigin,
                script_family: currentMetadata.scriptFamily,
                document_type: currentMetadata.documentType,
                notes: currentMetadata.notes,
              }
            : undefined,
        });
      let response: Awaited<ReturnType<typeof requestPage>>;
      try {
        response = await requestPage(reservedRunId);
      } catch (refusal: unknown) {
        // The reservation was refused (expired, already used, or unknown after a
        // backend restart): nothing was read. Read once more without one.
        const refused = reservedRunId && refusal instanceof ApiError && [404, 409, 410].includes(refusal.status);
        if (!refused) {
          throw refusal;
        }
        follower?.abort();
        follower = null;
        latestRunByPageRef.current[pageId] = null;
        patchRead({ progress: null, stage: "Reading the page" });
        response = await requestPage(null);
      }
      if (follower && String(response.run_id || "") !== reservedRunId) {
        // The backend did not use the reserved run, so its stream would never end.
        follower.abort();
        follower = null;
      }

      const rawExtractedText = mergeOCRResultText(response);
      const finalText = cleanTranscription(response, rawExtractedText);
      const storedText = finalText || "";
      // Marginalia, folio numbers and stamps are shown, but kept out of the text
      // that chat, search and entity extraction work from.
      const secondaryText = cleanTranscription(response, (response.secondary_lines ?? []).join("\n"));
      const runId = String(response.run_id || "").trim();

      // The pane and the viewer work from the lines themselves when the
      // engine reports them; otherwise from the text, one row per line.
      const fallbackBackends = new Set(
        (response.warnings ?? [])
          .map((warning) => /^LINES_READ_BY_FALLBACK:([^:]+):\d+$/.exec(String(warning))?.[1])
          .filter((name): name is string => Boolean(name)),
      );
      const bodyRegions = [...(response.line_regions ?? [])]
        .filter((region) => region.lane !== "secondary")
        .sort((a, b) => a.reading_order - b.reading_order);
      const textLines: TranscriptionLine[] = bodyRegions.length
        ? bodyRegions.map((region) => ({
            id: region.region_id,
            text: region.text,
            confidence: region.confidence,
            fallbackBackend: region.backend_name && fallbackBackends.has(region.backend_name) ? region.backend_name : null,
            onPage: Array.isArray(region.bbox_xyxy),
          }))
        : finalText
            .split("\n")
            .filter((line) => line.trim())
            .map((text, index) => ({ id: `line-${index}`, text, confidence: null, fallbackBackend: null, onPage: false }));
      setPageLinesByPageId((prev) => ({ ...prev, [pageId]: bodyRegions }));
      setTextLinesByPageId((prev) => ({ ...prev, [pageId]: textLines }));
      setMarginLinesByPageId((prev) => ({ ...prev, [pageId]: secondaryText.split("\n").filter((line) => line.trim()) }));
      if (response.original_image_width && response.original_image_height) {
        const size = { width: response.original_image_width, height: response.original_image_height };
        setPageSizeByPageId((prev) => ({ ...prev, [pageId]: size }));
      }
      setBadgesByPageId((prev) => ({ ...prev, [pageId]: describeExtraction(response, rawExtractedText) }));
      patchRead({ phase: "done", progress: null, stage: null, readBy: response.model_used || null });
      if (runId) {
        // A re-read replaces the run: its names are loaded afresh when asked for.
        setLinkingByRunId((prev) => ({ ...prev, [runId]: { phase: "idle" } }));
      }

      const authorityReport = String(
        response.consolidated_report
        || response.authority_report
        || response.mention_report
        || "",
      ).trim();
      if (!finalText) {
        // The page's state follows this read, not the one before it.
        setOcrTextByPageId((prev) => ({ ...prev, [pageId]: "" }));
        setOcrRunIdByPageId((prev) => ({ ...prev, [pageId]: runId }));
        setAuthorityReportByPageId((prev) => ({ ...prev, [pageId]: authorityReport }));
        const status = getExtractionStatus(response);
        setExtractionStatus(
          `Extraction complete (${status}): no readable text detected.`,
          describeExtraction(response, rawExtractedText),
        );
        if (!silent) {
          setMessagesByDocument((prev) => {
            const list = [...(prev[currentDocumentId] ?? [])];
            const index = list.findIndex((msg) => msg.id === assistantMessageId);
            if (index === -1) {
              return prev;
            }
            list[index] = { ...list[index], content: "No readable text detected on this page." };
            return { ...prev, [currentDocumentId]: list };
          });
        }
        return null;
      }

      setOcrTextByPageId((prev) => ({ ...prev, [currentPage.id]: storedText }));
      setOcrRunIdByPageId((prev) => ({ ...prev, [currentPage.id]: runId }));
      setAuthorityReportByPageId((prev) => ({ ...prev, [currentPage.id]: authorityReport }));

      if (!silent) {
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: (prev[currentDocumentId] ?? []).map((msg) =>
            msg.id === assistantMessageId
              ? {
                  ...msg,
                  content: `The transcription is in the Transcription pane: ${textLines.length} ${textLines.length === 1 ? "line" : "lines"}.`,
                }
              : msg,
          ),
        }));
      }
      const status = getExtractionStatus(response);
      const pipelineBits = [
        typeof response.chunks_count === "number" ? `${response.chunks_count} chunks` : "",
        typeof response.mentions_count === "number" ? `${response.mentions_count} mentions` : "",
      ].filter(Boolean);
      const readBy = response.model_used ? ` Read by ${response.model_used}.` : "";
      setExtractionStatus(
        pipelineBits.length
          ? `Extraction complete (${status}).${readBy} Knowledge pipeline: ${pipelineBits.join(", ")}.`
          : `Extraction complete (${status}).${readBy}`,
        describeExtraction(response, rawExtractedText),
      );
      return { text: storedText, runId, authorityReport };
    } catch (err: unknown) {
      let message = err instanceof Error ? err.message : "request failed";
      if (err instanceof ApiError && err.status === 413) {
        // The page travels as text inside the request, a third larger than the file; the server caps that.
        message = `${pageTooLargeMessage(currentPage.dataUrl.length)} ${message}`;
      }
      setExtractionStatus(`Extraction failed: ${message}`);
      patchRead({ phase: "failed", progress: null, stage: null, error: message });
      // A read that failed may never have claimed its run; its stream would stay open for an hour.
      follower?.abort();
      if (!silent) {
        setError(message);
      }
      if (!silent) {
        // The status message carries the failure; drop the empty answer bubble.
        setMessagesByDocument((prev) => ({
          ...prev,
          [currentDocumentId]: (prev[currentDocumentId] ?? []).filter(
            (msg) => msg.id !== assistantMessageId || msg.content,
          ),
        }));
      }
      return null;
    } finally {
      setSending(false);
      setAssistantLoadingLabel("Thinking");
    }
  };

  const handleTranslateInChat = async (userText: string, questionShown = false) => {
    if (!currentPage || !currentDocument) {
      return;
    }
    const userRequest = userText.trim() || "translate to english";
    let sourceText = String(currentExtractedText || "").trim();
    let pageContext: { transcript: string; runId: string; authorityReport: string } | undefined;
    if (!sourceText) {
      const extracted = await handleExtractTextInChat({ includeDebugOutput: false, silent: true });
      sourceText = String(extracted?.text || "").trim();
      if (extracted) {
        pageContext = { transcript: extracted.text, runId: extracted.runId, authorityReport: extracted.authorityReport };
      }
    }

    if (!sourceText) {
      appendMessages(currentDocument.id, [
        {
          id: makeId("msg-assistant-translate-empty"),
          role: "assistant",
          content: "No extracted text is available to translate.",
          createdAt: Date.now(),
        },
      ]);
      return;
    }

    const prompt = buildEnglishTranslationPrompt(userRequest, sourceText, currentMetadata);
    await sendPromptToChat(prompt, {
      displayText: userRequest,
      attachImage: false,
      loadingLabel: "Translating",
      modelOverride: taskModels.translation_model,
      chatStage: "translation",
      historyForModel: [],
      pageContext,
      questionShown,
    });
  };

  const handleEntityQuestionInChat = async (userText: string, questionShown = false) => {
    if (!currentPage || !currentDocument) {
      return;
    }
    let sourceText = String(currentExtractedText || "").trim();
    let authorityReport = String(currentAuthorityReport || "").trim();
    let runId = String(currentOcrRunId || "").trim();

    if (!sourceText || !runId) {
      const extracted = await handleExtractTextInChat({ includeDebugOutput: false, silent: true });
      sourceText = String(extracted?.text || sourceText || "").trim();
      authorityReport = String(extracted?.authorityReport || authorityReport || "").trim();
      runId = String(extracted?.runId || runId || "").trim();
    } else if (!authorityReport && runId) {
      try {
        const authorityPayload = await fetchAuthorityReport(runId);
        authorityReport = String(authorityPayload.report || "").trim();
        if (authorityReport) {
          setAuthorityReportByPageId((prev) => ({ ...prev, [currentPage.id]: authorityReport }));
        }
      } catch {
        authorityReport = "";
      }
    }

    if (!sourceText) {
      appendMessages(currentDocument.id, [
        {
          id: makeId("msg-assistant-entities-empty"),
          role: "assistant",
          content: "No extracted text is available for entity analysis.",
          createdAt: Date.now(),
        },
      ]);
      return;
    }
    const prompt = buildContextualUserPrompt(userText, {
      sourceText,
      authorityReport: authorityReport || "(no authority report available)",
      mode: "entities",
    });
    await sendPromptToChat(prompt, {
      displayText: userText,
      attachImage: false,
      loadingLabel: "Checking entities",
      chatStage: "entity_qa",
      historyForModel: [],
      pageContext: { transcript: sourceText, runId, authorityReport },
      questionShown,
    });
  };

  // Followers of finished or abandoned runs are stopped when the workspace closes.
  useEffect(() => {
    const followers = runFollowersRef.current;
    return () => {
      followers.forEach((controller) => controller.abort());
    };
  }, []);

  // On a phone the page comes first; wider screens start on its text.
  useEffect(() => {
    if (window.matchMedia("(max-width: 767px)").matches) {
      setPane("page");
    }
  }, []);

  const currentPageId = currentPage?.id ?? null;
  const currentReadState: ReadState = (() => {
    const stored = currentPageId ? readStateByPageId[currentPageId] : undefined;
    if (stored) {
      return stored;
    }
    // A page read before this session's pane state existed still has its text.
    return currentExtractedText ? { ...IDLE_READ_STATE, phase: "done" } : IDLE_READ_STATE;
  })();
  const currentTextLines: TranscriptionLine[] = useMemo(() => {
    const stored = currentPageId ? textLinesByPageId[currentPageId] : undefined;
    if (stored) {
      return stored;
    }
    return currentExtractedText
      .split("\n")
      .filter((line) => line.trim())
      .map((text, index) => ({ id: `line-${index}`, text, confidence: null, fallbackBackend: null, onPage: false }));
  }, [currentPageId, textLinesByPageId, currentExtractedText]);
  const currentLinking: LinkingState = currentOcrRunId ? linkingByRunId[currentOcrRunId] ?? { phase: "idle" } : { phase: "idle" };
  const reading = currentReadState.phase === "reading";
  const findingRegions = Boolean(currentPageId) && segmentingPageId === currentPageId;

  const loadLinking = (runId: string) => {
    setLinkingByRunId((prev) => ({ ...prev, [runId]: { phase: "loading" } }));
    fetchLinkingReport(runId)
      .then((report) => setLinkingByRunId((prev) => ({ ...prev, [runId]: { phase: "ready", report } })))
      .catch((err: unknown) =>
        setLinkingByRunId((prev) => ({
          ...prev,
          [runId]: { phase: "failed", error: err instanceof Error ? err.message : "The request failed." },
        })),
      );
  };

  // The names of a run are fetched when its panel is first opened.
  useEffect(() => {
    if (pane === "names" && currentOcrRunId && (linkingByRunId[currentOcrRunId]?.phase ?? "idle") === "idle") {
      loadLinking(currentOcrRunId);
    }
  }, [pane, currentOcrRunId, linkingByRunId]);

  const choosePages = () => {
    dialogOpenerRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    uploadInputRef.current?.click();
  };

  const readPage = () => {
    if (!currentPage || sending) {
      return;
    }
    setPane((value) => (value === "page" ? "text" : value));
    void handleExtractTextInChat({ silent: true });
  };

  const findRegions = async () => {
    if (!currentPage || findingRegions) {
      return;
    }
    const found = await runSegmentationForCurrentPage();
    if (found) {
      setShowSegmentationOverlay(true);
    }
    // The button had focus, was disabled while the regions were found, and is
    // replaced by the legend when they are: focus fell to the document body.
    // Put it on the legend, or back on the button if nothing was found.
    focusWhenShown("#pane-page details > summary, #pane-page [data-find-regions]");
  };

  const translate = () => {
    setPane("ask");
    void handleTranslateInChat("Translate this page into English.");
  };

  if (!clientReady) {
    return (
      <div className="flex h-dvh items-center justify-center bg-ground text-sm text-ink-muted" suppressHydrationWarning>
        Opening the workspace
      </div>
    );
  }

  const pageCount = currentDocument?.pages.length ?? 0;
  const metadataLine = currentMetadata
    ? [currentMetadata.language, currentMetadata.year, currentMetadata.placeOrOrigin, currentMetadata.scriptFamily, currentMetadata.documentType]
        .map((value) => String(value || "").trim())
        .filter(Boolean)
        .join(" · ")
    : "";
  const panel: "ask" | "names" | "run" | "about" = pane === "names" || pane === "run" || pane === "about" ? pane : "ask";
  const namesCount =
    currentLinking.phase === "ready" ? (currentLinking.report.mention_results ?? []).length : null;
  const tabs: { id: WorkspacePane; label: string; only?: string }[] = [
    { id: "page", label: "Page", only: "md:hidden" },
    { id: "text", label: "Transcription", only: "xl:hidden" },
    { id: "ask", label: "Ask" },
    { id: "names", label: namesCount === null ? "Names" : `Names ${namesCount}` },
    { id: "run", label: "Steps" },
    { id: "about", label: "Document" },
  ];
  const PANE_REGION: Record<WorkspacePane, string> = {
    page: "pane-page",
    text: "pane-text",
    ask: "pane-panel",
    names: "pane-panel",
    run: "pane-panel",
    about: "pane-panel",
  };
  // Left and right move between the tabs on show, as the tab pattern expects; only the selected tab is a tab stop.
  const onTabKey = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowRight" && event.key !== "ArrowLeft" && event.key !== "Home" && event.key !== "End") {
      return;
    }
    const shown = Array.from(event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')).filter(
      (tab) => tab.offsetParent !== null,
    );
    const at = shown.indexOf(document.activeElement as HTMLButtonElement);
    if (at === -1 || !shown.length) {
      return;
    }
    event.preventDefault();
    const next =
      event.key === "Home" ? 0 : event.key === "End" ? shown.length - 1 : (at + (event.key === "ArrowRight" ? 1 : -1) + shown.length) % shown.length;
    shown[next].focus();
    shown[next].click();
  };
  const tabButton = (tab: { id: WorkspacePane; label: string; only?: string }, selected: boolean, group: string) => (
    <button
      key={tab.id}
      id={`${group}-tab-${tab.id}`}
      type="button"
      role="tab"
      aria-selected={selected}
      aria-controls={PANE_REGION[tab.id]}
      tabIndex={selected ? 0 : -1}
      onClick={() => setPane(tab.id)}
      className={`relative -mb-px h-11 flex-none whitespace-nowrap border-b-2 px-1 text-sm outline-none focus-visible:ring-2 focus-visible:ring-brand ${
        selected ? "border-ink font-semibold text-ink" : "border-transparent text-ink-muted hover:text-ink"
      } ${tab.only ?? ""}`}
    >
      {tab.label}
    </button>
  );

  // aria-disabled, not disabled, at the first and last page: a button that is
  // disabled while it has focus drops a keyboard user on the document body.
  const pageNav =
    currentDocument && pageCount > 0 ? (
      <div className="flex items-center">
        <Button
          variant="quiet"
          className="w-8 px-0"
          aria-label="Previous page"
          aria-disabled={currentPageIndex <= 0 || undefined}
          onClick={() => {
            if (currentPageIndex > 0) {
              updateCurrentPageIndex(currentPageIndex - 1);
            }
          }}
        >
          <Icon name="chevron-left" />
        </Button>
        <select
          aria-label="Go to page"
          value={currentPageIndex}
          onChange={(event) => updateCurrentPageIndex(Number(event.target.value))}
          className="h-8 rounded-md bg-transparent px-1 text-sm tabular-nums outline-none hover:bg-active focus-visible:ring-2 focus-visible:ring-brand"
        >
          {currentDocument.pages.map((page, index) => (
            <option key={page.id} value={index}>
              Page {index + 1} of {pageCount}
            </option>
          ))}
        </select>
        <Button
          variant="quiet"
          className="w-8 px-0"
          aria-label="Next page"
          aria-disabled={currentPageIndex >= pageCount - 1 || undefined}
          onClick={() => {
            if (currentPageIndex < pageCount - 1) {
              updateCurrentPageIndex(currentPageIndex + 1);
            }
          }}
        >
          <Icon name="chevron-right" />
        </Button>
      </div>
    ) : null;

  // A fixed-height shell: its panes scroll, the page does not. Below 30rem of
  // height (a phone on its side, or a small window at 200% zoom) the panes
  // would be slits, so the shell keeps that height and the page scrolls.
  return (
    <div className="flex h-dvh min-h-[30rem] flex-col bg-ground text-ink">
      {/* A page change is otherwise silent for a screen reader: the buttons keep
          focus and the new page is an image. Announced when it changes. */}
      <p role="status" className="sr-only">
        {currentPage ? `Page ${currentPageIndex + 1} of ${pageCount}: ${currentPage.name}` : ""}
      </p>
      <input
        ref={uploadInputRef}
        type="file"
        accept="image/*"
        multiple
        className="sr-only"
        tabIndex={-1}
        aria-hidden="true"
        onChange={handleUpload}
      />

      {/* The frame: product, document and page on the left; the page's one job on the right. */}
      <header inert={pendingUpload ? true : undefined} className="flex h-12 flex-none items-center gap-3 px-4">
        <Mark />
        <span className="hidden text-sm font-semibold sm:inline">ArchAI</span>
        <span aria-hidden="true" className="hidden h-5 w-px bg-line sm:block" />
        <h1 className="min-w-0 truncate text-base font-semibold">{currentDocument ? currentDocument.name : "Manuscript workspace"}</h1>
        {documents.length > 1 ? (
          <select
            aria-label="Switch document"
            value={currentDocument?.id ?? ""}
            onChange={(event) => selectDocument(event.target.value)}
            className="h-8 max-w-40 rounded-md border border-field bg-sheet px-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-brand"
          >
            {documents.map((doc) => (
              <option key={doc.id} value={doc.id}>
                {doc.name}
              </option>
            ))}
          </select>
        ) : null}
        {metadataLine ? <p className="hidden min-w-0 truncate text-sm text-ink-muted lg:block">{metadataLine}</p> : null}

        <div className="ml-auto flex flex-none items-center gap-2">
          {pageNav ? <div className="hidden sm:block">{pageNav}</div> : null}
          <Button variant="secondary" data-open-pages onClick={choosePages}>
            <Icon name="upload" /> <span className="hidden sm:inline">Open pages</span>
            <span className="sr-only sm:hidden">Open pages</span>
          </Button>
          {currentPage ? (
            <Button variant="primary" onClick={readPage} disabled={sending} title={sending && !reading ? "Another request is still running" : undefined}>
              {reading ? "Reading" : currentTextLines.length ? "Read again" : "Read page"}
            </Button>
          ) : null}
        </div>
      </header>

      <div inert={pendingUpload ? true : undefined}>
        <BackendStatusBanner />
      </div>

      <div inert={pendingUpload ? true : undefined} className="flex min-h-0 flex-1 flex-col gap-3 px-2 pb-2 md:flex-row">
        {/* Pages of the document, on the frame. */}
        {currentDocument && pageCount > 1 ? (
          <nav aria-label="Pages" className="hidden w-[76px] flex-none overflow-y-auto md:block">
            <ol className="flex flex-col gap-2 px-1 py-1">
              {currentDocument.pages.map((page, index) => {
                const selected = index === currentPageIndex;
                return (
                  <li key={page.id}>
                    <button
                      type="button"
                      aria-label={`Page ${index + 1}: ${page.name}`}
                      aria-current={selected ? "page" : undefined}
                      onClick={() => updateCurrentPageIndex(index)}
                      className={`block w-full overflow-hidden rounded-md bg-sheet outline-none focus-visible:ring-2 focus-visible:ring-brand ${
                        selected ? "ring-2 ring-ink" : "ring-1 ring-line hover:ring-control"
                      }`}
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element -- a local data URL thumbnail */}
                      <img src={page.dataUrl} alt="" className="h-[84px] w-full object-cover object-top" />
                    </button>
                    <p className={`mt-0.5 text-center text-xs tabular-nums ${selected ? "text-ink" : "text-ink-muted"}`}>{index + 1}</p>
                  </li>
                );
              })}
            </ol>
          </nav>
        ) : null}

        {/* The page, on a well one step darker than the frame. */}
        <div id="pane-page" role="group" aria-label="Page" className={`min-h-0 min-w-0 flex-1 ${pane === "page" ? "flex" : "hidden"} flex-col md:flex`}>
          <div className="relative min-h-0 flex-1">
            {/* Before the viewer in the DOM, so Tab reaches it before the zoom bar at
                the bottom of the page, as it is placed; z-10 keeps it painted on top. */}
            {currentPage && !currentSegmentedPreview ? (
              <div className="absolute left-4 top-4 z-10">
                <Button data-find-regions variant="secondary" onClick={() => void findRegions()} disabled={findingRegions || sending}>
                  {findingRegions ? "Finding regions" : "Find layout regions"}
                </Button>
              </div>
            ) : null}
            <PageViewer
              page={currentPage}
              zoom={currentZoom}
              onZoom={updateZoom}
              regions={currentSegmentationLabels}
              showRegions={showSegmentationOverlay}
              onShowRegions={setShowSegmentationOverlay}
              lines={currentPageId ? pageLinesByPageId[currentPageId] ?? [] : []}
              pageSize={currentPageId ? pageSizeByPageId[currentPageId] ?? null : null}
              activeLineId={activeLineId}
              onActiveLine={setActiveLineId}
              onSelectLine={(id) => {
                setActiveLineId(id);
                setScrollToLineId(id);
                setPane((value) => (value === "page" ? "text" : value));
              }}
              onOpenPages={choosePages}
              onDropFiles={(files) => void openFiles(files)}
              openError={openError}
              pageNav={pageNav}
            />
            {currentSegmentationError ? (
              <p role="alert" className="absolute inset-x-4 top-14 rounded-md border border-line bg-sheet px-3 py-2 text-sm text-negative">
                Layout regions could not be found. {currentSegmentationError}
              </p>
            ) : null}
          </div>
        </div>

        {/* The work: the page's text, and beside it questions, names and steps. */}
        <div
          className={`order-first flex min-h-0 min-w-0 flex-col rounded-2xl border border-line bg-sheet md:order-none md:w-[440px] md:flex-none xl:w-[780px] ${
            pane === "page" ? "flex-none md:flex-none" : "flex-1 md:flex-none"
          }`}
        >
          <div
            role="tablist"
            aria-label="Workspace"
            onKeyDown={onTabKey}
            className="flex flex-none gap-5 overflow-x-auto border-b border-line px-6 [scrollbar-width:none] xl:hidden"
          >
            {tabs.map((tab) => tabButton(tab, pane === tab.id, "all"))}
          </div>
          <div className={`min-h-0 flex-1 ${pane === "page" ? "hidden md:flex" : "flex"}`}>
            <div id="pane-text" className={`min-h-0 min-w-0 flex-1 ${pane === "text" || pane === "page" ? "flex" : "hidden"} flex-col xl:flex`}>
              <TranscriptionPane
                pageName={currentPage?.name ?? null}
                state={currentReadState}
                lines={currentTextLines}
                marginLines={currentPageId ? marginLinesByPageId[currentPageId] ?? [] : []}
                badges={currentPageId ? badgesByPageId[currentPageId] ?? [] : []}
                activeLineId={activeLineId}
                scrollToLineId={scrollToLineId}
                onActiveLine={setActiveLineId}
                onRead={readPage}
                onTranslate={translate}
                busy={sending}
              />
            </div>
            <div
              className={`min-h-0 min-w-0 flex-1 flex-col xl:flex xl:w-[360px] xl:flex-none xl:border-l xl:border-line ${
                pane === "ask" || pane === "names" || pane === "run" || pane === "about" ? "flex" : "hidden"
              }`}
            >
              <div role="tablist" aria-label="Page panels" onKeyDown={onTabKey} className="hidden flex-none gap-5 border-b border-line px-6 xl:flex">
                {tabs.filter((tab) => !tab.only).map((tab) => tabButton(tab, panel === tab.id, "side"))}
              </div>
              <div id="pane-panel" role="tabpanel" aria-label={tabs.find((tab) => tab.id === panel)?.label} className="min-h-0 flex-1 overflow-y-auto">
                {panel === "ask" ? (
                  <ChatPanel
                    messages={currentMessages}
                    sending={sending}
                    asking={sending && !reading && !findingRegions}
                    loadingLabel={assistantLoadingLabel}
                    error={error}
                    input={input}
                    onInput={setInput}
                    onSend={() => void sendMessage()}
                    onClear={clearConversation}
                    disabledReason={!currentDocument ? "Open a document to ask about it" : !selectedModel ? "No chat model is available" : null}
                    announcement={replyAnnouncement}
                    hasTranscription={currentTextLines.length > 0}
                  />
                ) : null}
                {panel === "names" ? (
                  <EntitiesPanel
                    state={currentLinking}
                    hasRun={Boolean(currentOcrRunId)}
                    onRetry={() => currentOcrRunId && loadLinking(currentOcrRunId)}
                  />
                ) : null}
                {panel === "about" ? <DocumentPanel document={currentDocument} /> : null}
                {panel === "run" ? (
                  <RunPanel snapshot={currentPageId ? runSnapshotByPageId[currentPageId] ?? null : null} runId={currentOcrRunId || null} />
                ) : null}
              </div>
            </div>
          </div>
        </div>
      </div>

      {pendingUpload ? (
        <MetadataDialog
          draft={metadataDraft}
          pageCount={pendingUpload.pages.length}
          error={metadataError}
          onChange={setMetadataDraft}
          onCancel={cancelPendingUpload}
          onSave={commitPendingUpload}
        />
      ) : null}
    </div>
  );
}
