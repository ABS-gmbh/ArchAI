import type { Metadata } from "next";
import { Gentium_Plus, IBM_Plex_Sans } from "next/font/google";

import "./globals.css";

// The interface face. Its figures are tabular on request (font-variant-numeric).
const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin", "latin-ext"],
  weight: ["400", "500", "600"],
});

// The face for the manuscript's own words: it carries the medieval letters,
// abbreviation signs and combining marks that transcriptions contain.
const gentium = Gentium_Plus({
  variable: "--font-gentium",
  subsets: ["latin", "latin-ext"],
  weight: ["400", "700"],
  style: ["normal", "italic"],
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
