export interface ClockSink {
  sendTime(
    tPlay: number,
    playing: boolean,
    rate: number,
    seek: boolean,
    duration?: number,
  ): boolean;
}

/**
 * The browser video element is the playback clock. We report it at 4 Hz plus
 * on every play/pause/rate/seek transition. The backend projects between
 * reports and treats large jumps as discontinuities.
 */
export class PlayerClock {
  private readonly timer: number;
  private resumeAt = 0;
  private autoplay = true;
  private lastSentMs = 0;

  constructor(
    private readonly video: HTMLVideoElement,
    private readonly sink: ClockSink,
    private readonly onError?: (message: string) => void,
  ) {
    this.timer = window.setInterval(() => {
      if (!this.video.paused && !this.video.ended && this.video.readyState >= 2) {
        this.send(false);
      }
    }, 250);

    this.video.addEventListener("loadedmetadata", this.handleLoadedMetadata);
    this.video.addEventListener("play", this.handlePlay);
    this.video.addEventListener("pause", this.handlePause);
    this.video.addEventListener("seeked", this.handleSeeked);
    this.video.addEventListener("ratechange", this.handleRateChange);
    this.video.addEventListener("ended", this.handleEnded);
    this.video.addEventListener("error", this.handleError);
  }

  setSource(url: string, startAt = 0, autoplay = true): void {
    this.resumeAt = Math.max(0, startAt);
    this.autoplay = autoplay;
    this.video.src = url;
    this.video.load();
  }

  stop(): void {
    this.video.pause();
    this.video.removeAttribute("src");
    this.video.load();
    this.resumeAt = 0;
  }

  /** Send current state immediately (used after a WebSocket reconnect). */
  sync(seek = false): void {
    this.send(seek);
  }

  private send(seek: boolean): void {
    if (!this.video.currentSrc) return;
    const now = performance.now();
    if (!seek && now - this.lastSentMs < 180) return;
    this.lastSentMs = now;
    const duration = Number.isFinite(this.video.duration) ? this.video.duration : undefined;
    this.sink.sendTime(
      this.video.currentTime,
      !this.video.paused && !this.video.ended,
      this.video.playbackRate,
      seek,
      duration,
    );
  }

  private handleLoadedMetadata = (): void => {
    const duration = Number.isFinite(this.video.duration) ? this.video.duration : undefined;
    if (this.resumeAt > 0 && duration !== undefined) {
      this.video.currentTime = Math.min(this.resumeAt, Math.max(0, duration - 0.25));
    }
    this.send(true);
    if (this.autoplay) {
      void this.video.play().catch(() => {
        // Autoplay can be blocked after a reload; controls remain usable.
      });
    }
  };

  private handlePlay = (): void => {
    this.send(false);
  };

  private handlePause = (): void => {
    this.send(false);
  };

  private handleSeeked = (): void => {
    this.send(true);
  };

  private handleRateChange = (): void => {
    this.send(true);
  };

  private handleEnded = (): void => {
    this.send(false);
  };

  private handleError = (): void => {
    const error = this.video.error;
    const message = error ? `video error ${error.code}: ${error.message}` : "video error";
    this.onError?.(message);
  };

  dispose(): void {
    window.clearInterval(this.timer);
    this.video.removeEventListener("loadedmetadata", this.handleLoadedMetadata);
    this.video.removeEventListener("play", this.handlePlay);
    this.video.removeEventListener("pause", this.handlePause);
    this.video.removeEventListener("seeked", this.handleSeeked);
    this.video.removeEventListener("ratechange", this.handleRateChange);
    this.video.removeEventListener("ended", this.handleEnded);
    this.video.removeEventListener("error", this.handleError);
  }
}
