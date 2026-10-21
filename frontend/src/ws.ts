import type { ServerMessage } from "./types";

export type SocketStatus = "connecting" | "open" | "offline";

type MessageHandler = (message: ServerMessage) => void;
type StatusHandler = (status: SocketStatus) => void;
type OpenHandler = () => void;

/**
 * Resilient WebSocket client. Playback keeps working if the socket drops; the
 * player clock simply stops reaching the backend until reconnect.
 */
export class BackendSocket {
  private socket: WebSocket | null = null;
  private stopped = true;
  private retry = 0;
  private retryTimer: number | null = null;
  private pingTimer: number | null = null;
  private status: SocketStatus = "offline";
  private readonly messageHandlers = new Set<MessageHandler>();
  private readonly statusHandlers = new Set<StatusHandler>();
  private readonly openHandlers = new Set<OpenHandler>();

  connect(): void {
    this.stopped = false;
    this.open();
  }

  close(): void {
    this.stopped = true;
    this.clearTimers();
    if (this.socket) {
      this.socket.onclose = null;
      this.socket.close();
      this.socket = null;
    }
    this.setStatus("offline");
  }

  onMessage(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    handler(this.status);
    return () => this.statusHandlers.delete(handler);
  }

  onOpen(handler: OpenHandler): () => void {
    this.openHandlers.add(handler);
    return () => this.openHandlers.delete(handler);
  }

  sendTime(
    tPlay: number,
    playing: boolean,
    rate: number,
    seek: boolean,
    duration?: number,
  ): boolean {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(
      JSON.stringify({
        type: "time",
        t_play: tPlay,
        playing,
        rate,
        seek,
        duration: duration ?? null,
      }),
    );
    return true;
  }

  private open(): void {
    if (this.stopped) return;
    this.clearRetry();
    this.setStatus("connecting");
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${window.location.host}/ws`);
    this.socket = socket;

    socket.onopen = () => {
      this.retry = 0;
      this.setStatus("open");
      this.pingTimer = window.setInterval(() => {
        if (socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify({ type: "ping", t: Date.now() }));
        }
      }, 15_000);
      for (const handler of this.openHandlers) handler();
    };

    socket.onmessage = (event) => {
      let message: ServerMessage;
      try {
        message = JSON.parse(String(event.data)) as ServerMessage;
      } catch {
        console.warn("ignoring non-JSON websocket message");
        return;
      }
      for (const handler of this.messageHandlers) handler(message);
    };

    socket.onerror = () => {
      // onclose always follows; schedule reconnect there.
    };

    socket.onclose = () => {
      if (this.socket === socket) this.socket = null;
      if (this.pingTimer !== null) {
        window.clearInterval(this.pingTimer);
        this.pingTimer = null;
      }
      if (this.stopped) {
        this.setStatus("offline");
        return;
      }
      this.setStatus("offline");
      const delay = Math.min(5000, 500 * 2 ** this.retry);
      this.retry += 1;
      this.retryTimer = window.setTimeout(() => this.open(), delay);
    };
  }

  private clearTimers(): void {
    this.clearRetry();
    if (this.pingTimer !== null) {
      window.clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
  }

  private clearRetry(): void {
    if (this.retryTimer !== null) {
      window.clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
  }

  private setStatus(status: SocketStatus): void {
    if (this.status === status) return;
    this.status = status;
    for (const handler of this.statusHandlers) handler(status);
  }
}
