import { useState } from "react";
import type {
  GameWindowInfo,
  GlossaryHit,
  RuntimeMode,
  TranslationResult,
} from "./types/protocol";

type ViewMode = "wizard" | "overlay" | "studio";

const SAMPLE_WINDOWS: GameWindowInfo[] = [
  { window_id: "win-101", title: "Sakura Chronicle — Visual Novel", process_name: "sakura.exe" },
  { window_id: "win-204", title: "Hoshikuzu Demo (Wine)", process_name: "hoshikuzu.exe" },
];

const INITIAL_TRANSLATION: TranslationResult = {
  frame_id: "preview-001",
  source_text: "べ、別に桜ヶ丘学園の「星屑部」に誘いたいわけじゃないんだからね！",
  translated_text: "才、才不是想邀请你加入樱丘学园的“星屑部”呢！",
  speaker_resolved: "千夏",
  persona_id: "chinatsu (tsundere)",
  glossary_hits: [
    {
      source_term: "桜ヶ丘学園",
      target_term: "樱丘学园",
      category: "location",
      status: "verified",
      note: "已锁定译名",
    },
    {
      source_term: "星屑部",
      target_term: "星屑部",
      category: "term",
      status: "draft",
      note: "边翻边学自动提取（点击锁定/修改）",
    },
  ],
  telemetry: {
    vision_ms: 42.5,
    rag_ms: 3.1,
    prompt_ms: 0.8,
    ttft_ms: 310.0,
    llm_total_ms: 920.0,
    output_tokens: 24,
    decode_tps: 26.1,
  },
};

export function App() {
  const [view, setView] = useState<ViewMode>("wizard");
  const [profile, setProfile] = useState<RuntimeMode>("cpu_only");
  const [selectedWindow, setSelectedWindow] = useState<string>(SAMPLE_WINDOWS[0].window_id);
  const [hits, setHits] = useState<GlossaryHit[]>(INITIAL_TRANSLATION.glossary_hits);

  const handleToggleVerify = (sourceTerm: string) => {
    setHits((prev) =>
      prev.map((item) =>
        item.source_term === sourceTerm
          ? {
              ...item,
              target_term: item.source_term === "星屑部" ? "星屑社" : item.target_term,
              status: "verified",
              note: "已由用户在悬浮窗原位确认",
            }
          : item
      )
    );
  };

  return (
    <main className="shell">
      <header style={{ marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: "1.4rem" }}>KotoVN Desktop — Zero-Friction UI Scaffold</h1>
        <p style={{ margin: "6px 0 0", color: "#94a3b8", fontSize: "0.92rem" }}>
          本地推理优先（CPU / 12GB 混合） · OCR/VLM 自动读屏 · 边翻边学 RAG
        </p>
      </header>

      <nav className="nav" aria-label="视图切换">
        <button
          type="button"
          className={view === "wizard" ? "active" : ""}
          onClick={() => setView("wizard")}
        >
          1. 两步无感引导
        </button>
        <button
          type="button"
          className={view === "overlay" ? "active" : ""}
          onClick={() => setView("overlay")}
        >
          2. 悬浮字幕 HUD 预览
        </button>
        <button
          type="button"
          className={view === "studio" ? "active" : ""}
          onClick={() => setView("studio")}
        >
          3. Studio 抽屉预览
        </button>
      </nav>

      {view === "wizard" && (
        <section className="card">
          <h2 style={{ marginTop: 0 }}>Step 1：选择本地运行档位（无需手动调参）</h2>
          <div className="option-grid">
            <button
              type="button"
              className={`option-card ${profile === "cpu_only" ? "selected" : ""}`}
              onClick={() => setProfile("cpu_only")}
            >
              <strong>纯 CPU 省显存（推荐）</strong>
              <div style={{ fontSize: "0.84rem", color: "#94a3b8", marginTop: 6 }}>
                VRAM = 0 GB · GGUF Q4/IQ4 · ONNX OCR
              </div>
            </button>
            <button
              type="button"
              className={`option-card ${profile === "hybrid_12gb" ? "selected" : ""}`}
              onClick={() => setProfile("hybrid_12gb")}
            >
              <strong>12GB 显存混合加速</strong>
              <div style={{ fontSize: "0.84rem", color: "#94a3b8", marginTop: 6 }}>
                部分层卸载至 GPU（限额 &lt;= 6GB）· 支持按需 VLM
              </div>
            </button>
          </div>

          <h2 style={{ marginTop: 22 }}>Step 2：选择目标视觉小说窗口</h2>
          <div className="option-grid">
            {SAMPLE_WINDOWS.map((win) => (
              <button
                key={win.window_id}
                type="button"
                className={`option-card ${selectedWindow === win.window_id ? "selected" : ""}`}
                onClick={() => setSelectedWindow(win.window_id)}
              >
                <strong>{win.title}</strong>
                <div style={{ fontSize: "0.84rem", color: "#94a3b8", marginTop: 6 }}>
                  {win.process_name} · 自动检测对话框并隐藏主窗口
                </div>
              </button>
            ))}
          </div>

          <div style={{ marginTop: 20 }}>
            <button
              type="button"
              className="active"
              style={{
                padding: "10px 18px",
                borderRadius: 8,
                border: "none",
                background: "#3b82f6",
                color: "#fff",
                cursor: "pointer",
                fontWeight: 600,
              }}
              onClick={() => setView("overlay")}
            >
              附着窗口并进入无感悬浮字幕
            </button>
          </div>
        </section>
      )}

      {view === "overlay" && (
        <section className="hud-preview">
          <div className="speaker">
            {INITIAL_TRANSLATION.speaker_resolved} · {INITIAL_TRANSLATION.persona_id}
          </div>
          <p className="subtitle">
            {hits.some((h) => h.source_term === "星屑部" && h.target_term === "星屑社")
              ? "才、才不是想邀请你加入樱丘学园的“星屑社”呢！"
              : INITIAL_TRANSLATION.translated_text}
          </p>

          <div>
            {hits.map((hit) => (
              <button
                key={hit.source_term}
                type="button"
                className={`term-chip ${hit.status === "verified" ? "verified" : ""}`}
                onClick={() => handleToggleVerify(hit.source_term)}
                title={hit.note}
              >
                <span>
                  {hit.source_term} → {hit.target_term}
                </span>
                <span>[{hit.status}]</span>
              </button>
            ))}
          </div>

          <div className="meta-row">
            <span>Profile: {profile}</span>
            <span>Window: {selectedWindow}</span>
            <span>Vision: {INITIAL_TRANSLATION.telemetry.vision_ms} ms</span>
            <span>RAG: {INITIAL_TRANSLATION.telemetry.rag_ms} ms</span>
            <span>TTFT: {INITIAL_TRANSLATION.telemetry.ttft_ms} ms</span>
            <span>Speed: {INITIAL_TRANSLATION.telemetry.decode_tps} tok/s</span>
          </div>
        </section>
      )}

      {view === "studio" && (
        <section className="card">
          <h2 style={{ marginTop: 0 }}>Studio 开发者抽屉（热键 Alt+H 呼出）</h2>
          <p style={{ color: "#94a3b8", lineHeight: 1.6 }}>
            平时自动隐藏不打扰游玩。此处将在 Phase 3–4 承接：RAG 术语审校列表、读档 Checkpoint
            快照管理、人物人格卡片与实时试译对比框，以及量化模型显存/延迟监视器。
          </p>
        </section>
      )}
    </main>
  );
}
