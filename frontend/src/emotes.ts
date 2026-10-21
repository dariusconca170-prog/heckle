/**
 * Local emote set (M10). These are original SVGs shipped in `public/emotes`;
 * unknown 7TV/BTTV names stay as text, which is what a missing emote looks
 * like on Twitch too.
 */
export const EMOTES: Record<string, string> = {
  W: "w.svg",
  LUL: "lul.svg",
  KEKW: "kekw.svg",
  POG: "pog.svg",
  MONKAS: "monkas.svg",
  COPIUM: "copium.svg",
  RATIO: "ratio.svg",
  "<3": "heart.svg",
};

const LOOKUP = new Map(Object.entries(EMOTES).map(([token, file]) => [token.toUpperCase(), file]));

export function emoteFile(token: string): string | null {
  return LOOKUP.get(token.toUpperCase()) ?? null;
}

/**
 * Render chat text into a node, replacing known emote tokens with images.
 * Uses text nodes, never innerHTML, so generated chat cannot inject markup.
 */
export function renderChatText(container: HTMLElement, text: string): void {
  for (const token of String(text ?? "").split(/(\s+)/)) {
    if (!token) continue;
    if (/^\s+$/.test(token)) {
      container.append(document.createTextNode(token));
      continue;
    }
    const file = emoteFile(token);
    if (file) {
      const image = document.createElement("img");
      image.className = "chat-emote";
      image.src = `/emotes/${file}`;
      image.alt = token;
      image.title = token;
      container.append(image);
    } else {
      container.append(document.createTextNode(token));
    }
  }
}

const COLORS = [
  "#ff6b6b",
  "#ffa94d",
  "#ffd43b",
  "#a9e34b",
  "#63e6be",
  "#4dabf7",
  "#b197fc",
  "#f783ac",
  "#e599f7",
  "#66d9e8",
];

export function colorForHandle(handle: string): string {
  let hash = 0;
  for (let index = 0; index < handle.length; index += 1) {
    hash = (hash * 31 + handle.charCodeAt(index)) >>> 0;
  }
  return COLORS[hash % COLORS.length];
}

export function badgeFor(archetype: string): { label: string; className: string } | null {
  if (archetype === "lore-keeper") return { label: "mod", className: "mod" };
  if (archetype === "hype") return { label: "vip", className: "vip" };
  if (archetype === "wholesome") return { label: "sub", className: "sub" };
  return null;
}
