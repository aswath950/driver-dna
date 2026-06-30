import type { Metadata } from "next";
import { AppShell } from "@/components/AppShell";
import { readDisabledFeatures } from "@/lib/env";
import { enabledFeatures } from "@/lib/features";
import "./globals.css";

export const metadata: Metadata = {
  title: "Driver DNA",
  description: "F1 telemetry analytics dashboard",
};

// Read the operator section switches (FEATURE_* env vars) at request time, not build
// time. Without force-dynamic the layout could be statically prerendered, freezing
// the env values into the shell and re-breaking runtime toggling of the nav.
export const dynamic = "force-dynamic";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const features = enabledFeatures(readDisabledFeatures());

  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <AppShell features={features}>{children}</AppShell>
      </body>
    </html>
  );
}
