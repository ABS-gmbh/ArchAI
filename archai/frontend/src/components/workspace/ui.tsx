// Small shared pieces of the workspace's interface: one button, one pill, one
// icon set. Controls are 32px tall with a 6px radius; the one filled button is
// near-black, and the brand hue is kept for the line or region in focus.

import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonVariant = "primary" | "secondary" | "quiet";

const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: "bg-ink text-white hover:bg-ink/90 active:bg-ink/80 disabled:bg-control disabled:text-white",
  secondary: "border border-control bg-sheet text-ink hover:bg-well active:bg-active disabled:text-ink-muted",
  quiet: "text-ink hover:bg-active active:bg-control/60 disabled:text-ink-muted",
};

export function Button({
  variant = "secondary",
  className = "",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant }) {
  return (
    <button
      type="button"
      {...props}
      className={`inline-flex h-8 shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-md px-3 text-sm font-medium outline-none transition-colors focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-1 disabled:cursor-not-allowed ${BUTTON_VARIANTS[variant]} ${className}`}
    >
      {children}
    </button>
  );
}

export type PillTone = "positive" | "negative" | "warning" | "neutral";

const PILL_TONES: Record<PillTone, string> = {
  positive: "bg-pill-positive text-pill-positive-ink",
  negative: "bg-pill-negative text-pill-negative-ink",
  warning: "bg-pill-warning text-pill-warning-ink ring-1 ring-inset ring-[#fde68a]",
  neutral: "bg-pill-neutral text-pill-neutral-ink",
};

export function Pill({ tone = "neutral", title, children }: { tone?: PillTone; title?: string; children: ReactNode }) {
  return (
    <span
      title={title}
      className={`inline-flex h-5 flex-none items-center whitespace-nowrap rounded px-1.5 text-xs font-medium ${PILL_TONES[tone]}`}
    >
      {children}
    </span>
  );
}

/** 16px outline icons, 1.5px stroke, drawn on a 16 grid; decorative unless given a title. */
export function Icon({ name, className = "" }: { name: IconName; className?: string }) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 16 16"
      className={`h-4 w-4 flex-none ${className}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICONS[name]}
    </svg>
  );
}

export type IconName = "plus" | "minus" | "chevron-left" | "chevron-right" | "upload" | "retry" | "external" | "send";

const ICONS: Record<IconName, ReactNode> = {
  plus: <path d="M8 3.5v9M3.5 8h9" />,
  minus: <path d="M3.5 8h9" />,
  "chevron-left": <path d="M9.5 3.5 5 8l4.5 4.5" />,
  "chevron-right": <path d="M6.5 3.5 11 8l-4.5 4.5" />,
  upload: <path d="M8 10.5v-7M5 6l3-3 3 3M3 12.5h10" />,
  retry: <path d="M13 8a5 5 0 1 1-1.6-3.7M13 3v2.5h-2.5" />,
  external: <path d="M6.5 3.5h-3v9h9v-3M9 3.5h3.5V7M12.5 3.5 7.5 8.5" />,
  send: <path d="M3 8h9M8.5 4 12.5 8l-4 4" />,
};

/** The product's mark: a leaf of a book with its ruled lines, the top one in the brand hue. */
export function Mark({ className = "" }: { className?: string }) {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className={`h-6 w-6 flex-none ${className}`}>
      <path d="M5 2.5h10.5L19 6v15.5H5z" fill="var(--aa-sheet)" stroke="var(--aa-ink)" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M8 8.5h8" stroke="var(--aa-brand)" strokeWidth="2" strokeLinecap="round" />
      <path d="M8 12h8M8 15.5h8M8 19h4.5" stroke="var(--aa-ink)" strokeWidth="1.25" strokeLinecap="round" opacity="0.55" />
    </svg>
  );
}
