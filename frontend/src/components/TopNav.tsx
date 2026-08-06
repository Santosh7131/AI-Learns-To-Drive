import { Github, Moon, Sun } from "lucide-react";
import { useTheme } from "@/lib/theme";

export const REPO_URL = "https://github.com/Santosh7131/AI-Learns-To-Drive";

/** Minimal brand mark: a monochrome track loop with a moving dot. */
function Mark() {
  return (
    <div className="flex h-8 w-8 items-center justify-center rounded-lg border bg-card">
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" aria-hidden>
        <path
          d="M6 15c-2.2 0-4-1.6-4-3.6C2 8.8 4.6 7 8 7h7c2.8 0 5 1.6 5 3.8 0 2-1.8 3.2-4 3.2H6z"
          stroke="currentColor"
          strokeWidth="1.6"
        />
        <circle cx="8" cy="13.6" r="1.9" fill="hsl(var(--brand))" stroke="none" />
      </svg>
    </div>
  );
}

interface Props {
  right?: React.ReactNode;
  compact?: boolean;
}

export function TopNav({ right, compact }: Props) {
  const { theme, toggle } = useTheme();
  return (
    <header className="sticky top-0 z-30 border-b bg-background">
      <div className={`mx-auto flex h-14 items-center justify-between gap-3 px-4 sm:px-6 ${compact ? "" : "max-w-6xl"}`}>
        <a href="#top" className="flex items-center gap-2.5">
          <Mark />
          <div className="leading-none">
            <div className="text-sm font-semibold tracking-tight">AI Learns To Drive</div>
            {!compact && <div className="mt-0.5 hidden text-[11px] text-muted-foreground sm:block">Reinforcement learning · in your browser</div>}
          </div>
        </a>

        <div className="flex items-center gap-1.5">
          {right}
          <a
            href={REPO_URL}
            target="_blank"
            rel="noreferrer"
            title="View source on GitHub"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            <Github className="h-[18px] w-[18px]" />
          </a>
          <button
            onClick={toggle}
            title="Toggle theme"
            aria-label="Toggle theme"
            className="flex h-9 w-9 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
          >
            {theme === "dark" ? <Sun className="h-[18px] w-[18px]" /> : <Moon className="h-[18px] w-[18px]" />}
          </button>
        </div>
      </div>
    </header>
  );
}
