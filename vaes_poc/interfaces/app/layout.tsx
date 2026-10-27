import type { Metadata } from "next";
import "./globals.css";
import "./workspace.css";
import "./experience.css";
import "./queue-stage.css";
import "./models.css";

export const metadata: Metadata = {
  title: "VAE Lab · Triagem de raio-X",
  description:
    "Prova de conceito: triagem de raio-X de tórax com um autoencoder variacional",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="pt-BR">
      <body>{children}</body>
    </html>
  );
}
