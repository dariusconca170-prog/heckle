export interface MediaItem {
  id: string;
  name: string;
  size: number;
  modified: number;
  mime: string;
}

export interface SessionInfo {
  id: string;
  title: string;
  category: string;
  viewer_count: number;
  media: MediaItem;
  media_url: string;
  started_at: string;
  lookahead_s: number;
  duration: number | null;
}

export interface ClockState {
  t_play: number;
  t_read: number;
  playing: boolean;
  rate: number;
  duration: number | null;
  generation: number;
  lookahead_s: number;
  wall_time: number;
}

export interface MediaListResponse {
  items: MediaItem[];
  media_dir: string;
}

export interface SessionResponse {
  session: SessionInfo;
  clock: ClockState;
}

export interface SceneState {
  t: number;
  summary: string;
  action: string;
  objects: string[];
  mood: string;
  surprise: number;
  scene_change: boolean;
  on_screen_text: string;
  confidence: number;
  speech_text: string;
  speech_energy: number;
  source: string;
}

export interface TimelineEntry {
  scene: SceneState;
  received_at: number;
  generation: number;
  latency_s: number;
}

export interface SceneEvent {
  t: number;
  kind: string;
  intensity: number;
  detail: string;
}

export interface SceneSnapshot {
  current: SceneState | null;
  timeline: TimelineEntry[];
  events: SceneEvent[];
  sample_count: number;
  event_count: number;
  window_s: number;
  latency_s: {
    last: number | null;
    p50: number | null;
    p95: number | null;
  };
  updated_at: number | null;
}

export interface PerceptionMetrics {
  status: string;
  session_id: string | null;
  samples: number;
  errors: number;
  last_error: string;
  last_latency_s: number | null;
  last_t: number | null;
  next_sample_in_s: number;
  duty_cap: number;
}

export interface SceneResponse extends SceneSnapshot {
  active: boolean;
  service: PerceptionMetrics;
}

export interface SceneUpdatePayload {
  scene: SceneState;
  generation: number;
  latency_s: number;
  snapshot: SceneSnapshot;
  service: PerceptionMetrics;
  derived_events: SceneEvent[];
}

export interface SceneEventPayload {
  event: SceneEvent;
  scene: SceneState;
  snapshot: SceneSnapshot;
}

export interface DynamicsMetrics {
  status: string;
  session_id: string | null;
  buffer_level: number;
  target_buffer: number;
  low_buffer: number;
  critical_buffer: number;
  rate: number;
  intensity: number;
  excitation: number;
  cooling_down: boolean;
  active_chatters: number;
  released: number;
  generated: number;
  templates: number;
  copies: number;
  dropped: number;
  late_released: number;
  last_error: string;
  scene_t: number | null;
  next_refill_in_s: number;
}

export interface SystemStatusPayload {
  perception?: Partial<PerceptionMetrics> & { error?: string };
  dynamics?: Partial<DynamicsMetrics>;
}

export interface HelloMessage {
  type: "hello";
  config: {
    app_name: string;
    version: string;
    lookahead_s: number;
    time_update_hz: number;
  };
  session: SessionInfo | null;
  clock: ClockState;
  scene?: SceneSnapshot;
  perception?: PerceptionMetrics;
  dynamics?: DynamicsMetrics;
}

export interface BusEventMessage {
  type: "event";
  event_type: string;
  payload: Record<string, unknown>;
  ts: number;
  seq: number;
}

export interface PongMessage {
  type: "pong";
  t?: unknown;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type ServerMessage = HelloMessage | BusEventMessage | PongMessage | ErrorMessage;

export function isClockPayload(value: unknown): value is ClockState {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.t_play === "number" && typeof candidate.t_read === "number";
}

export interface ChatMessage {
  id: string;
  handle: string;
  archetype: string;
  text: string;
  source: string;
  batch_id: string;
  t: number;
  created_at: number;
  word_count: number;
  kind?: string;
  reply_to?: string | null;
  reply_to_handle?: string;
  reply_to_text?: string;
  mentions?: string[];
  emotes?: string[];
}

export interface Poll {
  id: number;
  stream_id: string;
  question: string;
  options: string[];
  counts: number[];
  percentages: number[];
  total_votes: number;
  opened_at: number;
  closes_at: number;
  duration_s: number;
  elapsed_s?: number | null;
  remaining_s?: number | null;
  allow_multiple: boolean;
  status: string;
  winner: number | null;
  reason?: string;
  winner_option?: string;
}

export interface PollsResponse {
  enabled: boolean;
  current: Poll | null;
  history: Poll[];
}

export interface ClipMessage {
  handle: string;
  text: string;
  t: number;
}

export interface Clip {
  id: number;
  stream_id: string;
  start_t: number;
  end_t: number;
  duration_s: number;
  score: number;
  peak_rate: number;
  z: number;
  reason: string;
  messages: ClipMessage[];
  path: string;
  thumb_path: string;
  status: string;
}

export interface ClipsResponse {
  enabled: boolean;
  clips: Clip[];
  metrics?: Record<string, unknown>;
}

export interface SystemSnapshot {
  gpu: {
    available: boolean;
    name: string;
    total_mb: number;
    used_mb: number;
    free_mb: number;
    utilization: number;
    error: string;
  };
  ram: {
    available: boolean;
    total_mb?: number;
    used_mb?: number;
    free_mb?: number;
    percent?: number;
  };
  chat: {
    enabled: boolean;
    last_tok_s: number;
    avg_tok_s: number;
    completions: number;
    last_error: string;
  };
  dynamics: {
    rate: number | null;
    buffer_level: number | null;
    target_buffer: number | null;
    active_chatters: number | null;
    replies?: number;
    reply_rate?: number;
    arguments?: number;
    mentions?: number;
  };
  viewers: number;
  session_id: string | null;
  clips: { clips?: number; exported?: number } | null;
  content_mode: string;
}

export interface PresetInfo {
  name: string;
  label: string;
  content_mode: string;
  intensity: number;
  chat_target_buffer: number;
  lookahead_s: number;
}

export interface SettingsSnapshot {
  preset: string;
  content_mode: string;
  intensity: number;
  chat_target_buffer: number;
  lookahead_s: number;
  model_path: string;
  model_quant: string;
  tts_enabled: boolean;
  mic_enabled: boolean;
  content_modes: string[];
  presets: PresetInfo[];
  revision: number;
}

export interface SettingsResponse {
  settings: SettingsSnapshot;
  effective: {
    lookahead_s: number;
    chat_target_buffer: number;
    mic_enabled: boolean;
    perception_enabled?: boolean;
  };
  model_profile?: { path: string; present: boolean };
}

export interface ChatBatchResponse {
  batch_id: string;
  scene: SceneState;
  messages: ChatMessage[];
  elapsed_s: number;
  fallback: boolean;
  error: string;
}

export interface ChatRecentResponse {
  messages: ChatMessage[];
}
