"use client";

import { useEffect, useMemo, useState } from "react";
import QRCode from "qrcode";

// Margem branca em módulos: menos que isso e alguns leitores de celular falham.
const MARGIN = 2;

/** QR que abre o site direto nesta aba, para a turma jogar pelo celular. */
export default function SiteQr() {
  // No deploy vem do domínio de produção; local, do endereço aberto no navegador.
  const [url, setUrl] = useState(
    process.env.SITE_URL ? `${process.env.SITE_URL}/#game` : null,
  );
  useEffect(() => {
    if (!url) setUrl(`${window.location.origin}${window.location.pathname}#game`);
  }, [url]);

  const cells = useMemo(() => {
    if (!url) return null;
    const { modules } = QRCode.create(url, { errorCorrectionLevel: "M" });
    const out: [number, number][] = [];
    for (let r = 0; r < modules.size; r++)
      for (let c = 0; c < modules.size; c++)
        if (modules.get(r, c)) out.push([r, c]);
    return { size: modules.size, out };
  }, [url]);

  const box = cells ? cells.size + 2 * MARGIN : 1;
  return (
    <>
      <a className="qr" href={url ?? "#game"} aria-label="QR code para abrir este desafio no celular">
        {cells && (
          <svg viewBox={`${-MARGIN} ${-MARGIN} ${box} ${box}`} shapeRendering="crispEdges" fill="#0b0e13">
            {cells.out.map(([r, c]) => (
              <rect key={`${r}-${c}`} x={c} y={r} width={1} height={1} />
            ))}
          </svg>
        )}
      </a>
      <p className="qr-note">{url?.replace(/^https?:\/\//, "")}</p>
    </>
  );
}
