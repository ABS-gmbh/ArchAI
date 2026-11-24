import type { Metadata } from "next";
import localFont from "next/font/local";

import "./globals.css";

// Every face is bundled (src/fonts, with its licence and source), so a build
// needs no access to Google Fonts.

// The interface face. Its figures are tabular on request (font-variant-numeric).
const plexSans = localFont({
  variable: "--font-plex-sans",
  src: [
    { path: "../fonts/ibm-plex-sans/IBMPlexSans-Regular.woff2", weight: "400", style: "normal" },
    { path: "../fonts/ibm-plex-sans/IBMPlexSans-Medium.woff2", weight: "500", style: "normal" },
    { path: "../fonts/ibm-plex-sans/IBMPlexSans-SemiBold.woff2", weight: "600", style: "normal" },
  ],
  display: "swap",
  fallback: ["system-ui", "sans-serif"],
});

// The face for the manuscript's own words: it carries the medieval letters,
// abbreviation signs and combining marks that transcriptions contain. These
// are SIL's complete files: Google's subsets leave out the combining letters
// (U+0363-U+036F), which then fell back to another face mid-word. Not
// preloaded (1.4 MB): a page fetches them once text in this face appears.
const gentium = localFont({
  variable: "--font-gentium",
  src: [
    { path: "../fonts/gentium-plus/GentiumPlus-Regular.woff2", weight: "400", style: "normal" },
    { path: "../fonts/gentium-plus/GentiumPlus-Italic.woff2", weight: "400", style: "italic" },
    { path: "../fonts/gentium-plus/GentiumPlus-Bold.woff2", weight: "700", style: "normal" },
    { path: "../fonts/gentium-plus/GentiumPlus-BoldItalic.woff2", weight: "700", style: "italic" },
  ],
  display: "swap",
  preload: false,
  // No fallbacks here: globals.css puts Junicode right after Gentium, and a
  // fallback in between (Times New Roman, Georgia) would draw some of the
  // characters Gentium lacks in a third face.
  adjustFontFallback: false,
  fallback: [],
});

// Second transcription face, for what Gentium Plus cannot draw: combining
// letters the recognisers output (U+1DE4 s above, U+1DD1 ur, U+1DE0 n...),
// ℥ ❧ ⁋ and MUFI private-use characters. A base letter and its mark that
// Gentium cannot draw together fall to Junicode as one cluster, so the mark
// sits on its letter. The release's variable Roman (weights 300-700),
// unmodified; scripts/check-font-coverage.mjs lists what each recogniser can
// output that neither face draws. Not preloaded (1.2 MB): fetched only when a
// page has such a character.
const junicode = localFont({
  variable: "--font-junicode",
  src: [{ path: "../fonts/junicode/JunicodeVF-Roman.woff2", weight: "300 700", style: "normal" }],
  display: "swap",
  preload: false,
  adjustFontFallback: false,
  fallback: [],
});

export const metadata: Metadata = {
  title: "ArchAI Workspace",
  description: "Document + chat workspace for manuscript analysis",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning className={`${plexSans.variable} ${gentium.variable} ${junicode.variable} font-sans antialiased`}>
        <main className="min-h-screen">{children}</main>
      </body>
    </html>
  );
}
