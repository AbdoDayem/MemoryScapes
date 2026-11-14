import type { ReactNode } from "react";

export const metadata = {
  title: "MemoryScapes",
  description: "Walk back through your memories, with the sound of the place around you.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
