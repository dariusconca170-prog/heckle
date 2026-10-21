import "./styles.css";
import { badgeFor, colorForHandle, renderChatText } from "./emotes";
import { PlayerClock } from "./player";
import type {
  ChatBatchResponse,
  ChatMessage,
  ChatRecentResponse,
  Clip,
  ClipsResponse,
  ClockState,
  DynamicsMetrics,
  MediaItem,
  MediaListResponse,
  PerceptionMetrics,
  Poll,
  PollsResponse,
  SceneEvent,
  SceneEventPayload,
  SceneResponse,
  SceneSnapshot,
  SceneUpdatePayload,
  ServerMessage,
  SessionInfo,
  SessionResponse,
  SettingsResponse,
  SettingsSnapshot,
  SystemSnapshot,
  SystemStatusPayload,
} from "./types";
import { isClockPayload } from "./types";
import { BackendSocket, type SocketStatus } from "./ws";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      // keep statusText
    }
    throw new ApiError(detail, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function byId<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
}

const setupScreen = byId<HTMLElement>("setup");
const studioScreen = byId<HTMLElement>("studio");
const mediaList = byId<HTMLDivElement>("media-list");
const refreshButton = byId<HTMLButtonElement>("refresh-media");
const startButton = byId<HTMLButtonElement>("start-stream");
const endButton = byId<HTMLButtonElement>("end-stream");
const titleInput = byId<HTMLInputElement>("stream-title");
const categoryInput = byId<HTMLInputElement>("stream-category");
const viewerInput = byId<HTMLInputElement>("viewer-count");
const setupError = byId<HTMLParagraphElement>("setup-error");
const video = byId<HTMLVideoElement>("player");
const playerPlaceholder = byId<HTMLDivElement>("player-placeholder");
const studioTitle = byId<HTMLElement>("studio-title");
const studioCategory = byId<HTMLElement>("studio-category");
const studioViewers = byId<HTMLElement>("studio-viewers");
const clockNote = byId<HTMLElement>("clock-note");
const wsStatus = byId<HTMLElement>("ws-status");
const chatSubtitle = byId<HTMLElement>("chat-subtitle");
const chatLog = byId<HTMLDivElement>("chat-log");
const chatIntro = byId<HTMLDivElement>("chat-intro");
const chatStatus = byId<HTMLElement>("chat-status");
const chatNote = byId<HTMLElement>("chat-note");
const generateChatButton = byId<HTMLButtonElement>("generate-chat");
const dynamicsStatus = byId<HTMLElement>("dynamics-status");
const dynamicsRate = byId<HTMLElement>("dynamics-rate");
const bufferMeter = byId<HTMLElement>("buffer-meter");
const bufferFill = byId<HTMLElement>("buffer-fill");
const dbgBuffer = byId<HTMLElement>("dbg-buffer");
const dbgIntensity = byId<HTMLElement>("dbg-intensity");
const dbgReleased = byId<HTMLElement>("dbg-released");
const dbgCopies = byId<HTMLElement>("dbg-copies");
const sceneStatus = byId<HTMLElement>("scene-status");
const sceneSummary = byId<HTMLElement>("scene-summary");
const sceneAction = byId<HTMLElement>("scene-action");
const sceneMood = byId<HTMLElement>("scene-mood");
const sceneSurprise = byId<HTMLElement>("scene-surprise");
const sceneConfidence = byId<HTMLElement>("scene-confidence");
const sceneTs = byId<HTMLElement>("scene-ts");
const sceneLatency = byId<HTMLElement>("scene-latency");
const sceneSamples = byId<HTMLElement>("scene-samples");
const sceneService = byId<HTMLElement>("scene-service");
const sceneText = byId<HTMLElement>("scene-text");
const sceneTimeline = byId<HTMLOListElement>("scene-timeline");
const sceneEvents = byId<HTMLUListElement>("scene-events");
const openSettingsButton = byId<HTMLButtonElement>("open-settings");
const closeSettingsButton = byId<HTMLButtonElement>("close-settings");
const settingsScreen = byId<HTMLElement>("settings");
const pollPanel = byId<HTMLElement>("poll-active");
const pollStatus = byId<HTMLElement>("poll-status");
const pollQuestion = byId<HTMLElement>("poll-question");
const pollTimer = byId<HTMLElement>("poll-timer");
const pollOptions = byId<HTMLDivElement>("poll-options");
const pollTotal = byId<HTMLElement>("poll-total");
const pollCloseButton = byId<HTMLButtonElement>("poll-close");
const pollCreateButton = byId<HTMLButtonElement>("poll-create-button");
const pollQuestionInput = byId<HTMLInputElement>("poll-question-input");
const pollOptionsInput = byId<HTMLTextAreaElement>("poll-options-input");
const pollDurationInput = byId<HTMLInputElement>("poll-duration");
const pollNote = byId<HTMLElement>("poll-note");
const clipList = byId<HTMLDivElement>("clip-list");
const clipNowButton = byId<HTMLButtonElement>("clip-now");
const clipNote = byId<HTMLElement>("clip-note");
const hudViewers = byId<HTMLElement>("hud-viewers");
const hudVram = byId<HTMLElement>("hud-vram");
const hudToks = byId<HTMLElement>("hud-toks");
const hudRam = byId<HTMLElement>("hud-ram");
const hudRate = byId<HTMLElement>("hud-rate");
const hudReplies = byId<HTMLElement>("hud-replies");
const presetRow = byId<HTMLDivElement>("preset-row");
const settingMode = byId<HTMLSelectElement>("setting-mode");
const settingIntensity = byId<HTMLInputElement>("setting-intensity");
const settingIntensityValue = byId<HTMLElement>("setting-intensity-value");
const settingBuffer = byId<HTMLInputElement>("setting-buffer");
const settingLookahead = byId<HTMLInputElement>("setting-lookahead");
const settingModel = byId<HTMLInputElement>("setting-model");
const settingQuant = byId<HTMLInputElement>("setting-quant");
const settingTts = byId<HTMLInputElement>("setting-tts");
const settingMic = byId<HTMLInputElement>("setting-mic");
const saveSettingsButton = byId<HTMLButtonElement>("save-settings");
const settingsStatus = byId<HTMLElement>("settings-status");
const deviceStatus = byId<HTMLElement>("device-status");
const micStartButton = byId<HTMLButtonElement>("mic-start");
const micStopButton = byId<HTMLButtonElement>("mic-stop");
const screenStartButton = byId<HTMLButtonElement>("screen-start");
const screenToggleButton = byId<HTMLButtonElement>("screen-toggle");
const screenStopButton = byId<HTMLButtonElement>("screen-stop");

let mediaItems: MediaItem[] = [];
let selectedId: string | null = null;
let currentSession: SessionInfo | null = null;
let socket: BackendSocket | null = null;
let lastClock: ClockState | null = null;
let lastScene: SceneSnapshot | null = null;
let sceneMetrics: PerceptionMetrics | null = null;
let dynamics: DynamicsMetrics | null = null;
let chatBusy = false;
const renderedChatIds = new Set<string>();
let updateCount = 0;
let noteTimer: number | null = null;
let currentPoll: Poll | null = null;
let clips: Clip[] = [];
let screenPaused = false;

const playerClock = new PlayerClock(
  video,
  {
    sendTime: (t, playing, rate, seek, duration) =>
      socket?.sendTime(t, playing, rate, seek, duration) ?? false,
  },
  (message) => setNote(message, 8000),
);

function setNote(text: string, holdMs = 3200): void {
  clockNote.textContent = text;
  if (noteTimer !== null) window.clearTimeout(noteTimer);
  noteTimer = window.setTimeout(() => {
    clockNote.textContent = currentSession ? "browser clock · 4 Hz" : "waiting for session";
  }, holdMs);
}

function showSetupError(message: string | null): void {
  setupError.hidden = message === null;
  setupError.textContent = message ?? "";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatTime(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const secs = Math.floor(safe % 60);
  const tenths = Math.floor((safe % 1) * 10);
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }
  return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenths}`;
}

function setWsStatus(status: SocketStatus): void {
  wsStatus.classList.remove("status-online", "status-offline", "status-connecting");
  if (status === "open") {
    wsStatus.textContent = "connected";
    wsStatus.classList.add("status-online");
  } else if (status === "connecting") {
    wsStatus.textContent = "connecting";
    wsStatus.classList.add("status-connecting");
  } else {
    wsStatus.textContent = "offline";
    wsStatus.classList.add("status-offline");
  }
}

function setClock(value: ClockState): void {
  lastClock = value;
  const tPlay = byId<HTMLElement>("dbg-tplay");
  const tRead = byId<HTMLElement>("dbg-tread");
  const generation = byId<HTMLElement>("dbg-generation");
  const lookahead = byId<HTMLElement>("dbg-lookahead");
  const playing = byId<HTMLElement>("dbg-playing");
  tPlay.textContent = `${formatTime(value.t_play)} (${value.t_play.toFixed(2)}s)`;
  tRead.textContent = `${formatTime(value.t_read)} (${value.t_read.toFixed(2)}s)`;
  generation.textContent = String(value.generation);
  lookahead.textContent = `${value.lookahead_s.toFixed(1)}s`;
  playing.textContent = value.playing ? "playing" : "paused";
  playing.classList.toggle("state-live", value.playing);
}

function sceneStatusText(snapshot: SceneSnapshot | null, metrics: PerceptionMetrics | null): string {
  const status = metrics?.status ?? "waiting";
  if (status === "error") {
    return `error · ${metrics?.last_error || "model unavailable"}`;
  }
  if (status === "paused") return "paused · look-ahead held";
  if (status === "priming") return "re-priming after clock change";
  if (status === "waiting") return "waiting for playback";
  if (status === "end") return "end of stream";
  if (status === "idle") return "idle · start a stream";
  if (snapshot?.current) {
    const latency = metrics?.last_latency_s ?? snapshot.latency_s.last;
    const latencyText = latency === null || latency === undefined ? "" : ` · ${latency.toFixed(2)}s`;
    return `live · ${snapshot.sample_count} sample${snapshot.sample_count === 1 ? "" : "s"}${latencyText}`;
  }
  return status === "disabled" ? "perception disabled" : "waiting for first frame";
}

function renderScene(snapshot: SceneSnapshot, metrics: PerceptionMetrics | null = sceneMetrics): void {
  lastScene = snapshot;
  if (metrics) sceneMetrics = metrics;
  sceneStatus.textContent = sceneStatusText(snapshot, sceneMetrics);
  sceneStatus.classList.toggle("state-warn", sceneMetrics?.status === "error");
  sceneService.textContent = sceneMetrics?.status ?? "—";
  sceneSamples.textContent = String(snapshot.sample_count);

  const current = snapshot.current;
  if (!current) {
    sceneSummary.textContent = "No frame observed yet.";
    sceneAction.textContent = "—";
    sceneMood.textContent = "—";
    sceneSurprise.textContent = "—";
    sceneConfidence.textContent = "—";
    sceneTs.textContent = "—";
    sceneLatency.textContent = "—";
    sceneText.hidden = true;
    renderSceneTimeline([]);
    renderSceneEvents([]);
    return;
  }

  sceneSummary.textContent = current.summary;
  sceneAction.textContent = current.action;
  sceneMood.textContent = current.mood;
  sceneSurprise.textContent = current.surprise.toFixed(2);
  sceneConfidence.textContent = current.confidence.toFixed(2);
  sceneTs.textContent = formatTime(current.t);
  const latency = sceneMetrics?.last_latency_s ?? snapshot.latency_s.last;
  sceneLatency.textContent =
    latency === null || latency === undefined ? "—" : `${latency.toFixed(2)}s`;
  if (current.on_screen_text.trim()) {
    sceneText.hidden = false;
    sceneText.textContent = `on screen: ${current.on_screen_text}`;
  } else {
    sceneText.hidden = true;
  }
  renderSceneTimeline(snapshot.timeline);
  renderSceneEvents(snapshot.events);
}

function renderSceneTimeline(entries: SceneSnapshot["timeline"]): void {
  sceneTimeline.replaceChildren();
  if (entries.length === 0) {
    const empty = document.createElement("li");
    empty.className = "scene-empty";
    empty.textContent = "waiting for the first observation";
    sceneTimeline.append(empty);
    return;
  }
  for (const entry of entries.slice(-8).reverse()) {
    const item = document.createElement("li");
    const stamp = document.createElement("span");
    stamp.className = "scene-stamp";
    stamp.textContent = formatTime(entry.scene.t);
    const text = document.createElement("span");
    text.className = "scene-line";
    text.textContent = `${entry.scene.action} · ${entry.scene.summary}`;
    item.append(stamp, text);
    sceneTimeline.append(item);
  }
}

function renderSceneEvents(events: SceneEvent[]): void {
  sceneEvents.replaceChildren();
  if (events.length === 0) {
    const empty = document.createElement("li");
    empty.className = "scene-empty";
    empty.textContent = "none yet";
    sceneEvents.append(empty);
    return;
  }
  for (const event of events.slice(-8).reverse()) {
    const item = document.createElement("li");
    const kind = document.createElement("span");
    kind.className = "scene-event-kind";
    kind.textContent = event.kind;
    const detail = document.createElement("span");
    detail.className = "scene-line";
    detail.textContent = `${formatTime(event.t)} · ${event.detail}`;
    item.append(kind, detail);
    sceneEvents.append(item);
  }
}

function renderScenePayload(payload: SceneUpdatePayload | SceneEventPayload): void {
  if (!payload.snapshot) return;
  const metrics = "service" in payload ? payload.service : sceneMetrics;
  renderScene(payload.snapshot, metrics);
}

function clearScene(): void {
  lastScene = null;
  sceneMetrics = null;
  renderScene(
    {
      current: null,
      timeline: [],
      events: [],
      sample_count: 0,
      event_count: 0,
      window_s: 90,
      latency_s: { last: null, p50: null, p95: null },
      updated_at: null,
    },
    null,
  );
  sceneStatus.textContent = "waiting for playback";
}

async function refreshScene(): Promise<void> {
  if (!currentSession) return;
  try {
    const response = await api<SceneResponse>("/api/scene");
    renderScene(response, response.service);
  } catch {
    // WebSocket scene updates remain the primary path; a failed poll is not
    // worth interrupting playback.
  }
}

const DEFAULT_DYNAMICS: DynamicsMetrics = {
  status: "idle",
  session_id: null,
  buffer_level: 0,
  target_buffer: 20,
  low_buffer: 8,
  critical_buffer: 3,
  rate: 0,
  intensity: 0.3,
  excitation: 0,
  cooling_down: false,
  active_chatters: 0,
  released: 0,
  generated: 0,
  templates: 0,
  copies: 0,
  dropped: 0,
  late_released: 0,
  last_error: "",
  scene_t: null,
  next_refill_in_s: 0,
};

function updateDynamics(metrics: Partial<DynamicsMetrics> | null): void {
  if (metrics === null) {
    dynamics = null;
  } else {
    dynamics = { ...(dynamics ?? DEFAULT_DYNAMICS), ...metrics };
  }
  const value = dynamics ?? DEFAULT_DYNAMICS;
  const target = Math.max(1, value.target_buffer);
  const level = Math.max(0, value.buffer_level);
  const percent = Math.min(100, (level / target) * 100);

  dynamicsStatus.textContent = `M3 · ${value.status}`;
  dynamicsStatus.classList.toggle("state-warn", value.status === "error");
  dynamicsRate.textContent = `${value.rate.toFixed(2)} msg/s`;
  dbgBuffer.textContent = `${level} / ${target}`;
  dbgIntensity.textContent = value.intensity.toFixed(2);
  dbgReleased.textContent = String(value.released);
  dbgCopies.textContent = String(value.copies);
  bufferFill.style.width = `${percent}%`;
  bufferMeter.setAttribute("aria-valuemax", String(target));
  bufferMeter.setAttribute("aria-valuenow", String(level));
  bufferMeter.classList.toggle("is-low", level < value.low_buffer);
  bufferMeter.classList.toggle("is-critical", level < value.critical_buffer);

  if (chatBusy) return;
  if (value.status === "error") {
    setChatBadge("degraded", "warn");
    chatNote.textContent = value.last_error || "chat degraded to templates";
  } else if (value.status === "paused") {
    setChatBadge("paused", "idle");
    chatNote.textContent = "paused · releases hold with the playhead";
  } else if (value.status === "priming") {
    setChatBadge("priming", "busy");
    chatNote.textContent = "priming · waiting for scene context";
  } else if (value.status === "live") {
    setChatBadge("live", "live");
    chatNote.textContent = `${value.generated} generated · ${value.templates} templates · ${value.dropped} dropped`;
  } else if (value.status === "end") {
    setChatBadge("ended", "idle");
    chatNote.textContent = "playhead reached the end of the file";
  } else if (currentSession) {
    setChatBadge("priming", "busy");
    chatNote.textContent = "Hawkes scheduler · waiting for first tick";
  } else {
    setChatBadge("idle", "idle");
    chatNote.textContent = "scheduler idle · start a stream";
  }
}

function setChatBadge(text: string, kind: "idle" | "busy" | "live" | "warn" = "idle"): void {
  chatStatus.textContent = text;
  chatStatus.classList.remove("state-live", "state-warn", "state-busy");
  if (kind === "live") chatStatus.classList.add("state-live");
  else if (kind === "warn") chatStatus.classList.add("state-warn");
  else if (kind === "busy") chatStatus.classList.add("state-busy");
}

function formatCount(value: number): string {
  return Math.max(0, Math.round(value)).toLocaleString();
}

function pollRemaining(poll: Poll): number {
  const tPlay = lastClock?.t_play ?? poll.opened_at;
  return Math.max(0, poll.duration_s - Math.max(0, tPlay - poll.opened_at));
}

function renderPoll(poll: Poll | null): void {
  currentPoll = poll;
  if (!poll) {
    pollPanel.hidden = true;
    pollStatus.textContent = "no poll";
    pollStatus.classList.remove("state-live", "state-warn");
    return;
  }
  pollPanel.hidden = false;
  pollQuestion.textContent = poll.question;
  const open = poll.status === "open";
  pollStatus.textContent = open ? "live" : "closed";
  pollStatus.classList.toggle("state-live", open);
  pollStatus.classList.toggle("state-warn", !open);
  pollCloseButton.hidden = !open;
  const timer = open ? pollRemaining(poll) : 0;
  pollTimer.textContent = open ? `${Math.ceil(timer)}s` : "results";
  pollTotal.textContent = `${formatCount(poll.total_votes)} vote${poll.total_votes === 1 ? "" : "s"}`;
  pollOptions.replaceChildren();
  const total = Math.max(1, poll.total_votes);
  const winner = poll.winner ?? null;
  poll.options.forEach((option, index) => {
    const count = poll.counts[index] ?? 0;
    const percent = poll.total_votes > 0 ? (count / total) * 100 : 0;
    const row = document.createElement("div");
    row.className = "poll-option";
    if (!open && winner === index) row.classList.add("is-winner");
    const bar = document.createElement("span");
    bar.className = "poll-bar";
    bar.style.width = `${percent}%`;
    const label = document.createElement("span");
    label.className = "poll-label";
    label.textContent = `${index + 1}. ${option}`;
    const value = document.createElement("span");
    value.className = "poll-value";
    value.textContent = `${percent.toFixed(0)}% · ${count}`;
    row.append(bar, label, value);
    if (open) {
      row.addEventListener("click", () => void voteInPoll(poll.id, index));
      row.title = "vote as you";
    } else {
      row.title = "poll closed";
    }
    pollOptions.append(row);
  });
}

async function voteInPoll(pollId: number, optionIndex: number): Promise<void> {
  try {
    const response = await api<{ poll: Poll }>(`/api/polls/${pollId}/vote`, {
      method: "POST",
      body: JSON.stringify({ handle: "you", option_index: optionIndex }),
    });
    renderPoll(response.poll);
    pollNote.textContent = "your vote was counted (once per poll)";
  } catch (error) {
    pollNote.textContent = error instanceof Error ? error.message : "vote failed";
  }
}

async function loadPolls(): Promise<void> {
  if (!currentSession) return;
  try {
    const response = await api<PollsResponse>("/api/polls");
    renderPoll(response.current);
    const closed = response.history.find((entry) => entry.status !== "open");
    pollNote.textContent = closed
      ? `last poll: ${closed.question} · ${closed.total_votes} votes`
      : "polls pause with the video";
  } catch {
    // Polls are optional furniture; a failed poll must not affect playback.
  }
}

async function createPoll(): Promise<void> {
  const question = pollQuestionInput.value.trim();
  const options = pollOptionsInput.value
    .split("\n")
    .map((option) => option.trim())
    .filter(Boolean);
  if (!question || options.length < 2) {
    pollNote.textContent = "write a question and at least two options";
    return;
  }
  pollCreateButton.disabled = true;
  try {
    const response = await api<{ poll: Poll }>("/api/polls", {
      method: "POST",
      body: JSON.stringify({
        question,
        options,
        duration_s: Number.parseInt(pollDurationInput.value, 10) || 60,
      }),
    });
    renderPoll(response.poll);
    pollNote.textContent = "open · chat starts voting";
    pollQuestionInput.value = "";
    pollOptionsInput.value = "";
  } catch (error) {
    pollNote.textContent = error instanceof Error ? error.message : "could not open poll";
  } finally {
    pollCreateButton.disabled = false;
  }
}

async function closePoll(): Promise<void> {
  pollCloseButton.disabled = true;
  try {
    const response = await api<{ poll: Poll }>("/api/polls/close", { method: "POST" });
    renderPoll(response.poll);
    pollNote.textContent = "closed · results announced in chat";
  } catch (error) {
    pollNote.textContent = error instanceof Error ? error.message : "could not close poll";
  } finally {
    pollCloseButton.disabled = false;
  }
}

function renderClips(): void {
  clipList.replaceChildren();
  if (clips.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No clips yet. A chat spike (z > 3) creates one automatically.";
    clipList.append(empty);
    return;
  }
  for (const clip of clips) {
    const card = document.createElement("article");
    card.className = "clip-card";
    if (clip.thumb_path) {
      const image = document.createElement("img");
      image.loading = "lazy";
      image.src = `/api/clips/${clip.id}/thumb`;
      image.alt = "clip thumbnail";
      card.append(image);
    }
    const body = document.createElement("div");
    body.className = "clip-body";
    const head = document.createElement("div");
    head.className = "clip-head";
    const stamp = document.createElement("strong");
    stamp.textContent = `${formatTime(clip.start_t)} – ${formatTime(clip.end_t)}`;
    const score = document.createElement("span");
    score.className = "muted";
    score.textContent = `z ${clip.z.toFixed(1)} · ${clip.score.toFixed(0)}`;
    head.append(stamp, score);
    const reason = document.createElement("p");
    reason.className = "clip-reason";
    reason.textContent = clip.reason;
    body.append(head, reason);
    if (clip.path) {
      const link = document.createElement("a");
      link.className = "text-button";
      link.href = `/api/clips/${clip.id}/file`;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = "open clip";
      body.append(link);
    } else {
      const pending = document.createElement("span");
      pending.className = "muted";
      pending.textContent = clip.status === "failed" ? "export failed" : "exporting…";
      body.append(pending);
    }
    card.append(body);
    clipList.append(card);
  }
}

async function loadClips(): Promise<void> {
  if (!currentSession) return;
  try {
    const response = await api<ClipsResponse>("/api/clips?limit=30");
    clips = response.clips ?? [];
    renderClips();
    if (clips.length > 0) {
      clipNote.textContent = `${clips.filter((clip) => clip.path).length} exported · ${clips.length} total`;
    }
  } catch {
    // Clips are derived furniture; never block the studio on them.
  }
}

async function clipNow(): Promise<void> {
  if (!currentSession) return;
  clipNowButton.disabled = true;
  try {
    const response = await api<{ clip: Clip }>("/api/clips", { method: "POST", body: "{}" });
    clipNote.textContent = `clip queued · ${response.clip.reason}`;
    await loadClips();
    window.setTimeout(() => void loadClips(), 4000);
  } catch (error) {
    clipNote.textContent = error instanceof Error ? error.message : "clip failed";
  } finally {
    clipNowButton.disabled = false;
  }
}

function updateSystemHud(system: SystemSnapshot): void {
  hudViewers.textContent = formatCount(system.viewers || currentSession?.viewer_count || 0);
  if (system.gpu.available) {
    hudVram.textContent = `${system.gpu.used_mb} / ${system.gpu.total_mb} MB`;
    hudVram.title = `${system.gpu.name} · ${system.gpu.utilization}%`;
  } else {
    hudVram.textContent = "no GPU";
    hudVram.title = system.gpu.error;
  }
  hudToks.textContent =
    system.chat.last_tok_s > 0 ? `${system.chat.last_tok_s.toFixed(1)}` : system.chat.enabled ? "—" : "off";
  hudRam.textContent = system.ram.available
    ? `${system.ram.percent?.toFixed(0)}% · ${formatBytes((system.ram.used_mb ?? 0) * 1024 * 1024)}`
    : "—";
  hudRate.textContent = `${(system.dynamics.rate ?? 0).toFixed(2)}/s`;
  hudReplies.textContent = `${system.dynamics.replies ?? 0} (${((system.dynamics.reply_rate ?? 0) * 100).toFixed(0)}%)`;
  if (system.viewers > 0) studioViewers.textContent = `${formatCount(system.viewers)} viewers`;
}

async function refreshSystem(): Promise<void> {
  try {
    const response = await api<SystemSnapshot>("/api/system");
    updateSystemHud(response);
  } catch {
    // The HUD is diagnostic only.
  }
}

function setSettingsStatus(text: string, warn = false): void {
  settingsStatus.textContent = text;
  settingsStatus.classList.toggle("state-warn", warn);
}

function fillSettings(data: SettingsResponse): void {
  settingMode.value = data.settings.content_mode;
  settingIntensity.value = String(data.settings.intensity);
  settingIntensityValue.textContent = data.settings.intensity.toFixed(1);
  settingBuffer.value = String(data.settings.chat_target_buffer);
  settingLookahead.value = String(data.settings.lookahead_s);
  settingModel.value = data.settings.model_path;
  settingQuant.value = data.settings.model_quant;
  settingTts.checked = data.settings.tts_enabled;
  settingMic.checked = data.settings.mic_enabled;
  presetRow.replaceChildren();
  for (const preset of data.settings.presets) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "preset-button";
    button.textContent = preset.label;
    button.setAttribute("aria-pressed", String(preset.name === data.settings.preset));
    button.addEventListener("click", () => void applyPreset(preset.name));
    presetRow.append(button);
  }
  setSettingsStatus(
    `revision ${data.settings.revision} · look-ahead applies to the next stream`,
  );
}

async function loadSettings(): Promise<void> {
  try {
    const response = await api<SettingsResponse>("/api/settings");
    fillSettings(response);
  } catch (error) {
    setSettingsStatus(error instanceof Error ? error.message : "could not load settings", true);
  }
}

async function applyPreset(name: string): Promise<void> {
  try {
    await api<{ settings: SettingsSnapshot }>(`/api/settings/preset/${name}`, { method: "POST" });
    await loadSettings();
    setSettingsStatus(`preset: ${name}`);
  } catch (error) {
    setSettingsStatus(error instanceof Error ? error.message : "preset failed", true);
  }
}

async function saveSettings(): Promise<void> {
  saveSettingsButton.disabled = true;
  try {
    const response = await api<SettingsResponse>("/api/settings", {
      method: "PATCH",
      body: JSON.stringify({
        content_mode: settingMode.value,
        intensity: Number.parseFloat(settingIntensity.value),
        chat_target_buffer: Number.parseInt(settingBuffer.value, 10),
        lookahead_s: Number.parseFloat(settingLookahead.value),
        model_path: settingModel.value.trim(),
        model_quant: settingQuant.value.trim(),
        tts_enabled: settingTts.checked,
        mic_enabled: settingMic.checked,
      }),
    });
    fillSettings(response);
    setSettingsStatus("saved · intensity and content mode apply now");
  } catch (error) {
    setSettingsStatus(error instanceof Error ? error.message : "could not save settings", true);
  } finally {
    saveSettingsButton.disabled = false;
  }
}

async function refreshDevices(): Promise<void> {
  try {
    const [mic, screen] = await Promise.all([
      api<{ status: string; last_text?: string; enabled?: boolean }>("/api/mic"),
      api<{ status: string; paused?: boolean; backend?: string }>("/api/screen"),
    ]);
    screenPaused = Boolean(screen.paused);
    deviceStatus.textContent = `mic ${mic.status}${mic.last_text ? ` · heard “${mic.last_text}”` : ""} · screen ${screen.status}${screen.backend ? ` (${screen.backend})` : ""}`;
    screenToggleButton.textContent = screenPaused ? "resume capture" : "pause capture";
  } catch {
    deviceStatus.textContent = "device status unavailable";
  }
}

async function deviceAction(path: string, label: string): Promise<void> {
  try {
    await api(path, { method: "POST" });
    deviceStatus.textContent = label;
    await refreshDevices();
  } catch (error) {
    deviceStatus.textContent = error instanceof Error ? error.message : `${label} failed`;
  }
}

function addChatMessage(message: ChatMessage): void {
  if (renderedChatIds.has(message.id)) return;
  renderedChatIds.add(message.id);
  chatIntro.hidden = true;
  const pinned = chatLog.scrollTop + chatLog.clientHeight >= chatLog.scrollHeight - 48;
  const line = document.createElement("div");
  line.className = "chat-line";
  line.dataset.archetype = message.archetype;

  if (message.kind === "system" || message.archetype === "system") {
    line.classList.add("chat-system");
    const dot = document.createElement("span");
    dot.className = "system-accent";
    dot.textContent = "◆";
    const text = document.createElement("span");
    text.className = "chat-text";
    text.textContent = message.text;
    line.append(dot, text);
    chatLog.append(line);
    if (pinned) chatLog.scrollTop = chatLog.scrollHeight;
    return;
  }

  if (message.reply_to && message.reply_to_text) {
    const quote = document.createElement("div");
    quote.className = "chat-quote";
    const who = document.createElement("b");
    who.textContent = `@${message.reply_to_handle || "viewer"}`;
    quote.append(who, document.createTextNode(` ${message.reply_to_text}`));
    line.append(quote);
  }

  const badge = badgeFor(message.archetype);
  if (badge) {
    const tag = document.createElement("span");
    tag.className = `chat-badge-icon ${badge.className}`;
    tag.textContent = badge.label;
    line.append(tag);
  }

  const user = document.createElement("span");
  user.className = "chat-user";
  user.textContent = `${message.handle}:`;
  user.style.color = colorForHandle(message.handle);
  user.title = message.archetype;
  const text = document.createElement("span");
  text.className = "chat-text";
  renderChatText(text, message.text);
  line.append(user, text);
  if (message.source !== "model" && message.source !== "crowd") {
    line.dataset.source = message.source;
  }
  chatLog.append(line);
  if (pinned) chatLog.scrollTop = chatLog.scrollHeight;
}

function resetChatLog(): void {
  renderedChatIds.clear();
  chatLog.replaceChildren(chatIntro);
  chatIntro.hidden = false;
  if (currentSession) {
    setChatBadge("live", "live");
    chatNote.textContent = "Hawkes scheduler · waiting for first tick";
  } else {
    setChatBadge("idle", "idle");
    chatNote.textContent = "waiting for session";
  }
}

async function refreshRecentChat(): Promise<void> {
  if (!currentSession) return;
  try {
    const response = await api<ChatRecentResponse>("/api/chat/recent?limit=50");
    if (response.messages.length === 0) return;
    resetChatLog();
    for (const message of response.messages) addChatMessage(message);
    chatLog.scrollTop = chatLog.scrollHeight;
    if (dynamics) updateDynamics(dynamics);
    else setChatBadge("live", "live");
  } catch {
    // Chat is a live feature; a failed refresh must not break playback.
  }
}

async function generateChatNow(): Promise<void> {
  if (chatBusy) return;
  chatBusy = true;
  generateChatButton.disabled = true;
  generateChatButton.textContent = "generating…";
  setChatBadge("thinking", "busy");
  chatNote.textContent = "one batched 4B call · filtering refusals and captions";
  try {
    const response = await api<ChatBatchResponse>("/api/chat/generate", {
      method: "POST",
      body: JSON.stringify({ count: 10 }),
    });
    for (const message of response.messages) addChatMessage(message);
    const fallbackCount = response.messages.filter((message) => message.source !== "model").length;
    if (response.fallback) {
      setChatBadge("templates", "warn");
      chatNote.textContent = `${response.elapsed_s.toFixed(2)}s · model unavailable, used templates`;
    } else if (fallbackCount > 0) {
      setChatBadge(`${response.messages.length} lines`, "warn");
      chatNote.textContent = `${response.elapsed_s.toFixed(2)}s · ${fallbackCount} filtered → template`;
    } else {
      setChatBadge(`${response.messages.length} lines`, "live");
      chatNote.textContent = `${response.elapsed_s.toFixed(2)}s · scene t=${formatTime(response.scene.t)} · all model`;
    }
  } catch (error) {
    setChatBadge("error", "warn");
    chatNote.textContent = error instanceof Error ? error.message : "chat generation failed";
  } finally {
    chatBusy = false;
    generateChatButton.disabled = !currentSession;
    generateChatButton.textContent = "Preview 10 messages (bypasses timing)";
  }
}

function handleServerMessage(message: ServerMessage): void {
  if (message.type === "hello") {
    setClock(message.clock);
    setNote("websocket connected");
    if (message.scene) {
      renderScene(message.scene, message.perception ?? null);
    }
    updateDynamics(message.dynamics ?? null);
    void loadPolls();
    void loadClips();
    void refreshSystem();
    void refreshDevices();
    if (message.session && currentSession && message.session.id !== currentSession.id) {
      currentSession = message.session;
      applySessionMeta(message.session);
    }
    return;
  }
  if (message.type === "error") {
    setNote(`server: ${message.message}`, 6000);
    return;
  }
  if (message.type === "pong") return;
  if (message.type !== "event") return;

  if (message.event_type.startsWith("clock.")) {
    updateCount += 1;
    if (isClockPayload(message.payload)) {
      setClock(message.payload);
    }
    if (message.event_type === "clock.seek") {
      setNote("seek detected · queued reactions flushed");
      if (sceneMetrics) sceneMetrics.status = "priming";
      if (lastScene) renderScene(lastScene, sceneMetrics);
    } else if (message.event_type === "clock.pause") {
      setNote("paused · look-ahead primed for resume");
    } else if (message.event_type === "clock.play") {
      setNote("resumed · re-priming look-ahead");
    }
  } else if (message.event_type === "chat.message") {
    addChatMessage(message.payload as unknown as ChatMessage);
    if (!chatBusy) updateDynamics({ status: "live" });
  } else if (message.event_type === "chat.buffer_level") {
    updateDynamics(message.payload as unknown as Partial<DynamicsMetrics>);
  } else if (message.event_type === "chat.catching_up") {
    if (!chatBusy) {
      setChatBadge("catching up", "busy");
      chatNote.textContent = "catching up · buffer below target";
    }
  } else if (message.event_type === "poll.opened") {
    const payload = message.payload as unknown as { poll: Poll };
    renderPoll(payload.poll);
    pollNote.textContent = "open · chat is voting";
  } else if (message.event_type === "poll.vote") {
    const payload = message.payload as unknown as {
      poll_id: number;
      counts: number[];
      percentages: number[];
      total_votes: number;
    };
    if (currentPoll && currentPoll.id === payload.poll_id) {
      currentPoll = {
        ...currentPoll,
        counts: payload.counts,
        percentages: payload.percentages,
        total_votes: payload.total_votes,
      };
      renderPoll(currentPoll);
    }
  } else if (message.event_type === "poll.closed" || message.event_type === "poll.results") {
    const payload = message.payload as unknown as { poll: Poll };
    renderPoll(payload.poll);
    pollNote.textContent = payload.poll.winner_option
      ? `${payload.poll.winner_option} won · chat is reacting`
      : "closed · no votes";
  } else if (message.event_type === "stream.event") {
    const payload = message.payload as unknown as {
      viewer_count?: number;
      event?: { kind: string; detail?: string };
    };
    if (payload.viewer_count) {
      studioViewers.textContent = `${formatCount(payload.viewer_count)} viewers`;
      hudViewers.textContent = formatCount(payload.viewer_count);
    }
    if (payload.event?.kind && payload.event.kind !== "viewers") {
      setNote(payload.event.detail || payload.event.kind, 5000);
    }
  } else if (message.event_type === "clip.candidate") {
    clipNote.textContent = "clip candidate · exporting after the segment plays";
    window.setTimeout(() => void loadClips(), 6000);
  } else if (message.event_type === "clip.created") {
    void loadClips();
    clipNote.textContent = "clip exported · preview below";
  } else if (message.event_type === "mic.speech") {
    const payload = message.payload as unknown as { text: string };
    deviceStatus.textContent = `heard “${payload.text}” · chat is reacting`;
  } else if (message.event_type === "scene.update") {
    renderScenePayload(message.payload as unknown as SceneUpdatePayload);
  } else if (message.event_type === "scene.event") {
    renderScenePayload(message.payload as unknown as SceneEventPayload);
  } else if (message.event_type === "system.status") {
    const payload = message.payload as unknown as SystemStatusPayload;
    if (payload.dynamics) updateDynamics(payload.dynamics);
    if (payload.perception) {
      sceneMetrics = {
        status: payload.perception.status ?? sceneMetrics?.status ?? "idle",
        session_id: payload.perception.session_id ?? sceneMetrics?.session_id ?? null,
        samples: payload.perception.samples ?? sceneMetrics?.samples ?? 0,
        errors: payload.perception.errors ?? sceneMetrics?.errors ?? 0,
        last_error: payload.perception.last_error ?? payload.perception.error ?? sceneMetrics?.last_error ?? "",
        last_latency_s: payload.perception.last_latency_s ?? sceneMetrics?.last_latency_s ?? null,
        last_t: payload.perception.last_t ?? sceneMetrics?.last_t ?? null,
        next_sample_in_s: payload.perception.next_sample_in_s ?? sceneMetrics?.next_sample_in_s ?? 0,
        duty_cap: payload.perception.duty_cap ?? sceneMetrics?.duty_cap ?? 0.35,
      };
      if (lastScene) renderScene(lastScene, sceneMetrics);
      else {
        sceneStatus.textContent = sceneStatusText(null, sceneMetrics);
        sceneService.textContent = sceneMetrics.status;
      }
    }
  } else if (message.event_type === "session.ended") {
    setNote("session ended");
    resetToSetup();
  }
}

function applySessionMeta(session: SessionInfo): void {
  studioTitle.textContent = session.title;
  studioCategory.textContent = session.category || "no category";
  studioViewers.textContent = `${session.viewer_count.toLocaleString()} viewers`;
  chatSubtitle.textContent = `watching ${session.media.name}`;
  document.title = `${session.title} · Twitch Simulator`;
}

function openSettings(): void {
  settingsScreen.hidden = false;
  setupScreen.hidden = true;
  studioScreen.hidden = true;
  void loadSettings();
  void refreshDevices();
}

function closeSettings(): void {
  settingsScreen.hidden = true;
  setupScreen.hidden = currentSession !== null;
  studioScreen.hidden = currentSession === null;
}

function startStudio(session: SessionInfo, clock: ClockState, autoplay: boolean): void {
  currentSession = session;
  lastClock = clock;
  updateCount = 0;
  setupScreen.hidden = true;
  studioScreen.hidden = false;
  playerPlaceholder.hidden = true;
  applySessionMeta(session);
  setClock(clock);
  clearScene();
  resetChatLog();
  updateDynamics(null);
  currentPoll = null;
  clips = [];
  renderPoll(null);
  renderClips();
  void loadPolls();
  void loadClips();
  void refreshSystem();
  generateChatButton.disabled = false;
  generateChatButton.textContent = "Preview 10 messages (bypasses timing)";

  socket?.close();
  socket = new BackendSocket();
  socket.onStatus(setWsStatus);
  socket.onMessage(handleServerMessage);
  socket.onOpen(() => {
    playerClock.sync(false);
    setNote("websocket connected");
    void refreshScene();
    void refreshRecentChat();
  });
  socket.connect();

  setNote(autoplay ? "starting playback" : "restored paused session");
  playerClock.setSource(session.media_url, clock.t_play, autoplay);
}

function resetToSetup(): void {
  currentSession = null;
  socket?.close();
  socket = null;
  playerClock.stop();
  studioScreen.hidden = true;
  setupScreen.hidden = false;
  playerPlaceholder.hidden = false;
  document.title = "Twitch Simulator";
  lastClock = null;
  lastScene = null;
  sceneMetrics = null;
  updateCount = 0;
  setWsStatus("offline");
  clockNote.textContent = "waiting for session";
  clearScene();
  resetChatLog();
  updateDynamics(null);
  currentPoll = null;
  clips = [];
  renderPoll(null);
  renderClips();
  generateChatButton.disabled = true;
  void loadMedia();
}

function renderMedia(): void {
  mediaList.replaceChildren();
  if (mediaItems.length === 0) {
    const empty = document.createElement("div");
    empty.className = "media-empty";
    empty.innerHTML =
      "<strong>No videos found.</strong><span>Drop an mp4/mkv/webm into <code>data/media</code> and refresh.</span>";
    mediaList.append(empty);
    startButton.disabled = true;
    return;
  }

  for (const item of mediaItems) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "media-option";
    button.dataset.id = item.id;
    button.setAttribute("aria-pressed", String(item.id === selectedId));

    const name = document.createElement("span");
    name.className = "media-name";
    name.textContent = item.name;

    const meta = document.createElement("span");
    meta.className = "media-meta";
    meta.textContent = `${formatBytes(item.size)} · ${item.id}`;

    button.append(name, meta);
    button.addEventListener("click", () => selectMedia(item));
    mediaList.append(button);
  }
  if (selectedId) {
    const stillExists = mediaItems.some((item) => item.id === selectedId);
    if (!stillExists) selectedId = null;
  }
  startButton.disabled = selectedId === null;
}

function selectMedia(item: MediaItem): void {
  const previous = mediaItems.find((candidate) => candidate.id === selectedId);
  selectedId = item.id;
  if (!titleInput.value.trim() || (previous && titleInput.value === previous.name)) {
    titleInput.value = item.name;
  }
  for (const button of mediaList.querySelectorAll<HTMLButtonElement>(".media-option")) {
    button.setAttribute("aria-pressed", String(button.dataset.id === item.id));
  }
  startButton.disabled = false;
  showSetupError(null);
}

async function loadMedia(): Promise<void> {
  refreshButton.disabled = true;
  refreshButton.textContent = "scanning…";
  try {
    const response = await api<MediaListResponse>("/api/media");
    mediaItems = response.items;
    renderMedia();
    showSetupError(null);
  } catch (error) {
    showSetupError(error instanceof Error ? error.message : "failed to scan media");
  } finally {
    refreshButton.disabled = false;
    refreshButton.textContent = "refresh";
  }
}

async function startStream(): Promise<void> {
  if (!selectedId) {
    showSetupError("pick a video first");
    return;
  }
  startButton.disabled = true;
  startButton.textContent = "starting…";
  showSetupError(null);
  try {
    const response = await api<SessionResponse>("/api/session", {
      method: "POST",
      body: JSON.stringify({
        media_id: selectedId,
        title: titleInput.value.trim(),
        category: categoryInput.value.trim(),
        viewer_count: Number.parseInt(viewerInput.value, 10) || 0,
      }),
    });
    startStudio(response.session, response.clock, true);
  } catch (error) {
    showSetupError(error instanceof Error ? error.message : "failed to start session");
  } finally {
    startButton.disabled = selectedId === null;
    startButton.textContent = "Start stream";
  }
}

async function endStream(): Promise<void> {
  if (!currentSession) return;
  endButton.disabled = true;
  try {
    await api<{ ended: string | null }>("/api/session", { method: "DELETE" });
  } catch (error) {
    setNote(error instanceof Error ? error.message : "failed to end session", 6000);
  } finally {
    endButton.disabled = false;
    resetToSetup();
  }
}

async function bootstrap(): Promise<void> {
  refreshButton.addEventListener("click", () => void loadMedia());
  startButton.addEventListener("click", () => void startStream());
  endButton.addEventListener("click", () => void endStream());
  generateChatButton.addEventListener("click", () => void generateChatNow());
  openSettingsButton.addEventListener("click", openSettings);
  closeSettingsButton.addEventListener("click", closeSettings);
  saveSettingsButton.addEventListener("click", () => void saveSettings());
  settingIntensity.addEventListener("input", () => {
    settingIntensityValue.textContent = Number.parseFloat(settingIntensity.value).toFixed(1);
  });
  pollCreateButton.addEventListener("click", () => void createPoll());
  pollCloseButton.addEventListener("click", () => void closePoll());
  clipNowButton.addEventListener("click", () => void clipNow());
  micStartButton.addEventListener("click", () => void deviceAction("/api/mic/start", "microphone starting"));
  micStopButton.addEventListener("click", () => void deviceAction("/api/mic/stop", "microphone stopped"));
  screenStartButton.addEventListener("click", () => void deviceAction("/api/screen/start", "screen capture starting"));
  screenToggleButton.addEventListener("click", () => void deviceAction("/api/screen/toggle", "capture toggled"));
  screenStopButton.addEventListener("click", () => void deviceAction("/api/screen/stop", "screen capture stopped"));
  viewerInput.addEventListener("change", () => {
    const value = Number.parseInt(viewerInput.value, 10);
    if (!Number.isFinite(value) || value < 0) viewerInput.value = "0";
  });
  window.addEventListener("beforeunload", () => socket?.close());
  window.setInterval(() => {
    if (currentSession) void refreshScene();
  }, 5000);
  window.setInterval(() => {
    if (currentSession && settingsScreen.hidden) void refreshSystem();
  }, 2000);
  window.setInterval(() => {
    if (currentSession) void loadClips();
  }, 15000);

  updateDebugLoop();
  await loadMedia();

  try {
    const response = await api<SessionResponse>("/api/session");
    startStudio(response.session, response.clock, response.clock.playing);
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 404) {
      showSetupError(error instanceof Error ? error.message : "failed to restore session");
    }
  }
}

function updateDebugLoop(): void {
  const tPlay = byId<HTMLElement>("dbg-tplay");
  const tRead = byId<HTMLElement>("dbg-tread");
  const drift = byId<HTMLElement>("dbg-drift");
  const generation = byId<HTMLElement>("dbg-generation");
  const lookahead = byId<HTMLElement>("dbg-lookahead");
  const playing = byId<HTMLElement>("dbg-playing");
  const videoInfo = byId<HTMLElement>("dbg-video");
  const updates = byId<HTMLElement>("dbg-updates");

  window.setInterval(() => {
    if (!currentSession) return;
    if (lastClock) {
      tPlay.textContent = `${formatTime(lastClock.t_play)} (${lastClock.t_play.toFixed(2)}s)`;
      tRead.textContent = `${formatTime(lastClock.t_read)} (${lastClock.t_read.toFixed(2)}s)`;
      generation.textContent = String(lastClock.generation);
      lookahead.textContent = `${lastClock.lookahead_s.toFixed(1)}s`;
      playing.textContent = video.paused ? "paused" : "playing";
      playing.classList.toggle("state-live", !video.paused);
      const local = video.currentTime;
      const delta = local - lastClock.t_play;
      drift.textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}s`;
      drift.classList.toggle("state-warn", Math.abs(delta) > 1.0);
    }
    if (video.videoWidth > 0) {
      videoInfo.textContent = `${video.videoWidth}×${video.videoHeight} · ${formatTime(video.duration)}`;
    } else {
      videoInfo.textContent = "—";
    }
    updates.textContent = String(updateCount);
    if (currentPoll?.status === "open") {
      pollTimer.textContent = `${Math.ceil(pollRemaining(currentPoll))}s`;
    }
  }, 250);
}

void bootstrap();
