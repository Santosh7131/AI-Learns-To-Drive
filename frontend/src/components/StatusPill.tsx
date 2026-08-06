/** small labelled pill for the status strip (minimal, solid) */
export function Pill({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`inline-flex items-center gap-1.5 rounded-md border bg-card px-2 py-1 text-xs ${className}`}>
      {children}
    </div>
  );
}
