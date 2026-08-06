/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional base URL of a separately-hosted backend (e.g. https://api.example.com). */
  readonly VITE_API_URL?: string;
  /** "true"/"1" forces the in-browser playback playground (static deploy, no backend probe). */
  readonly VITE_PLAYBACK_ONLY?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
