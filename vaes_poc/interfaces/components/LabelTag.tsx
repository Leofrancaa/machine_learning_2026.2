import type { Label } from "@/lib/types";

/** O laudo de um exame, ou "oculto" enquanto a plateia não deve vê-lo. */
export default function LabelTag({ label, hidden = false }: { label: Label; hidden?: boolean }) {
  return hidden ? <span className="label hidden">oculto</span> : <span className={`label ${label}`}>{label}</span>;
}
