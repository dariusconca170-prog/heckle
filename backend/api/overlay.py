"""OBS browser-source chat overlay (M10).

This is a standalone page, not part of the Vite app: OBS should keep working
even if the frontend build is stale. It renders only chat, on a transparent
background, with no scrollbar and no chrome. Add it as a Browser Source at
400x800 (or 400x900) in OBS.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_EMOTES = {
    "W": "w.svg",
    "LUL": "lul.svg",
    "KEKW": "kekw.svg",
    "POG": "pog.svg",
    "MONKAS": "monkas.svg",
    "COPIUM": "copium.svg",
    "RATIO": "ratio.svg",
    "<3": "heart.svg",
}

_OVERLAY = """<!doctype html>
<!-- OBS Browser Source: url /overlay/chat, width 400, height 800, transparent background. -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Twitch Sim overlay</title>
<style>
  :root { color-scheme: dark; }
  html, body { margin: 0; padding: 0; background: transparent !important; overflow: hidden; }
  body {
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    font-size: clamp(16px, 2.4vw, 22px);
    color: #efeff1;
    text-shadow: 0 1px 2px rgba(0,0,0,.95), 0 0 4px rgba(0,0,0,.8);
    width: 100vw; height: 100vh;
  }
  #wrap { position: fixed; inset: 0; display: flex; flex-direction: column; justify-content: flex-end; }
  #log {
    overflow-y: auto; overflow-x: hidden; scrollbar-width: none;
    max-height: 100%; padding: 8px 12px;
    display: flex; flex-direction: column; gap: 3px;
  }
  #log::-webkit-scrollbar { display: none; }
  .line { line-height: 1.35; word-wrap: break-word; animation: enter 120ms ease-out; }
  @keyframes enter { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
  .user { font-weight: 700; }
  .badge {
    display: inline-block; font-size: .62em; font-weight: 800; line-height: 1;
    padding: 2px 4px; margin-right: 4px; border-radius: 3px; vertical-align: 1px;
    text-shadow: none; letter-spacing: .02em;
  }
  .badge.sub { background: #3b2a63; color: #dcc7ff; }
  .badge.mod { background: #0e5c36; color: #b9f5d0; }
  .badge.vip { background: #6b1747; color: #ffc2e2; }
  .quote { display: block; border-left: 2px solid #772ce8; padding-left: 6px; margin: 1px 0 2px;
           color: #b9b9c1; font-size: .82em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .quote b { color: #d6b8ff; font-weight: 600; }
  .system { color: #c7a5ff; font-weight: 600; }
  .system .dot { color: #9147ff; }
  .emote { height: 1.45em; width: auto; vertical-align: -.32em; margin: 0 1px; }
  #status { position: fixed; right: 8px; bottom: 6px; font-size: 10px; opacity: .35; }
</style>
</head>
<body>
<div id="wrap"><div id="log" aria-live="polite"></div></div>
<div id="status">connecting</div>
<script>
(() => {
  const EMOTES = __EMOTES__;
  const log = document.getElementById("log");
  const status = document.getElementById("status");
  const seen = new Set();
  const colors = ["#ff6b6b","#ffa94d","#ffd43b","#a9e34b","#63e6be","#4dabf7","#b197fc","#f783ac","#e599f7","#66d9e8"];
  const badges = { "lore-keeper": ["mod","mod"], "hype": ["vip","vip"], "wholesome": ["sub","sub"] };
  const colorFor = (name) => {
    let hash = 0;
    for (let i = 0; i < name.length; i += 1) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
    return colors[hash % colors.length];
  };
  const emoteFor = (token) => {
    const key = token.toUpperCase();
    return EMOTES[key] || (token === "<3" ? EMOTES["<3"] : "");
  };
  const renderText = (node, text) => {
    for (const token of String(text).split(/(\\s+)/)) {
      if (!token) continue;
      if (/^\\s+$/.test(token)) { node.append(document.createTextNode(token)); continue; }
      const file = emoteFor(token);
      if (file) {
        const img = document.createElement("img");
        img.className = "emote";
        img.src = "/emotes/" + file;
        img.alt = token;
        node.append(img);
      } else {
        node.append(document.createTextNode(token));
      }
    }
  };
  const line = (message) => {
    if (!message || !message.id || seen.has(message.id)) return;
    seen.add(message.id);
    if (seen.size > 400) seen.delete(seen.values().next().value);
    const row = document.createElement("div");
    row.className = "line";
    if (message.kind === "system" || message.archetype === "system") {
      row.classList.add("system");
      const dot = document.createElement("span");
      dot.className = "dot";
      dot.textContent = "◆ ";
      row.append(dot, document.createTextNode(message.text || ""));
      log.append(row);
      trim();
      return;
    }
    if (message.reply_to && message.reply_to_text) {
      const quote = document.createElement("span");
      quote.className = "quote";
      const who = document.createElement("b");
      who.textContent = "@" + (message.reply_to_handle || "viewer");
      quote.append(who, document.createTextNode(" " + message.reply_to_text));
      row.append(quote);
    }
    const badge = badges[message.archetype];
    if (badge) {
      const tag = document.createElement("span");
      tag.className = "badge " + badge[0];
      tag.textContent = badge[1];
      row.append(tag);
    }
    const user = document.createElement("span");
    user.className = "user";
    user.style.color = colorFor(message.handle || "viewer");
    user.textContent = (message.handle || "viewer") + ": ";
    const text = document.createElement("span");
    renderText(text, message.text || "");
    row.append(user, text);
    log.append(row);
    trim();
  };
  const trim = () => {
    while (log.childElementCount > 160) log.firstElementChild.remove();
    log.scrollTop = log.scrollHeight;
  };
  const connect = () => {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws`);
    ws.onopen = () => { status.textContent = ""; };
    ws.onclose = () => { status.textContent = "reconnecting"; setTimeout(connect, 1500); };
    ws.onerror = () => { status.textContent = "offline"; };
    ws.onmessage = (event) => {
      let data;
      try { data = JSON.parse(event.data); } catch { return; }
      if (data.type === "event" && data.event_type === "chat.message") line(data.payload);
    };
    setInterval(() => {
      if (ws.readyState === 1) ws.send(JSON.stringify({ type: "ping" }));
    }, 20000);
  };
  fetch("/api/chat/recent?limit=40")
    .then((response) => response.json())
    .then((body) => { for (const message of body.messages || []) line(message); })
    .catch(() => {})
    .finally(connect);
})();
</script>
</body>
</html>
"""


@router.get("/overlay/chat", response_class=HTMLResponse, include_in_schema=False)
@router.get("/overlay", response_class=HTMLResponse, include_in_schema=False)
def overlay_chat() -> HTMLResponse:
    import json

    return HTMLResponse(_OVERLAY.replace("__EMOTES__", json.dumps(_EMOTES)))


__all__ = ["router"]
