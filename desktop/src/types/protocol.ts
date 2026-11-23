/**
 * Shared IPC protocol types synchronized with backend/src/kotovn/schemas/.
 */

export type RuntimeMode = "cpu_only" | "hybrid_12gb" | "research_gpu";

export type EntryStatus = "draft" | "verified";

export interface GlossaryHit {
  source_term: string;
  target_term: string;
  category: string;
  status: EntryStatus;
  note: string;
}

export interface PipelineTelemetry {
  vision_ms: number;
  rag_ms: number;
  prompt_ms: number;
  ttft_ms: number;
  llm_total_ms: number;
  output_tokens: number;
  decode_tps: number;
}

export interface TranslationResult {
  frame_id: string;
  source_text: string;
  translated_text: string;
  speaker_resolved: string | null;
  persona_id: string | null;
  glossary_hits: GlossaryHit[];
  telemetry: PipelineTelemetry;
}

export interface GameWindowInfo {
  window_id: string;
  title: string;
  process_name: string;
}
