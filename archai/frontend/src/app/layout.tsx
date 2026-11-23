import type { Metadata } from "next";
import localFont from "next/font/local";

import "./globals.css";

// Both faces are bundled (src/fonts, with their licences and sources), so a
// build needs no access to Google Fonts.

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
  adjustFontFallback: "Times New Roman",
  fallback: ["Georgia", "serif"],
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
      <body suppressHydrationWarning className={`${plexSans.variable} ${gentium.variable} font-sans antialiased`}>
        <main className="min-h-screen">{children}</main>
      </body>
    </html>
  );
}
