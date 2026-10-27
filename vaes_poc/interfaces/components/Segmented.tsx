interface Option<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

/** Grupo de botões em que só um fica ligado. */
export default function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
  title,
}: {
  options: Option<T>[];
  value: T;
  onChange: (value: T) => void;
  label: string;
  title?: string;
}) {
  const disabled = options.some((o) => o.disabled);
  return (
    <div className={`segmented ${disabled ? "disabled" : ""}`} role="group" aria-label={label} title={title}>
      {options.map((o) => (
        <button
          key={o.value}
          className={o.value === value ? "on" : ""}
          aria-pressed={o.value === value}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
