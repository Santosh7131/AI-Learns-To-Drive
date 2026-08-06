/** small labelled pill for the header status bar (shared by live + playback shells) */
export function Pill({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`glass-hud flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs ${className}`}>
      {children}
    </div>
  );
}
