// env.d.ts
interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
  // add any other VITE_* env vars here
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}