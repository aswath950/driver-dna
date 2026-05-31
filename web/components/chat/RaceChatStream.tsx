"use client";

import { useRef, useState } from "react";
import { Button } from "@/components/ui/Button";
import { env } from "@/lib/env";

type Event =
  | { type: "token"; delta: string }
  | { type: "tool_call"; tool: string; args: Record<string, unknown> }
  | { type: "tool_result"; tool: string; summary: string }
  | { type: "done"; input_tokens: number; output_tokens: number; rounds: number }
  | { type: "error"; detail: string };

/**
 * SSE client for POST /api/v1/ai/race-chat/stream.
 *
 * We can't use `EventSource` because it's GET-only — we POST a JSON body
 * and parse the response stream manually.
 */
export function RaceChatStream({ sessionId }: { sessionId: number }) {
  const [message, setMessage] = useState("");
  const [events, setEvents] = useState<Event[]>([]);
  const [running, setRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  async function ask() {
    if (!message.trim() || running) return;
    setRunning(true);
    setEvents([]);
    const ac = new AbortController();
    abortRef.current = ac;

    try {
      const res = await fetch(`${env.NEXT_PUBLIC_API_BASE}/api/v1/ai/race-chat/stream`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ session_id: sessionId, message }),
        signal: ac.signal,
      });
      if (!res.body) {
        setEvents([{ type: "error", detail: "No response body" }]);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE events are separated by a blank line.
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          const ev = parseSSE(block);
          if (ev) setEvents((prev) => [...prev, ev]);
        }
      }
    } catch (e) {
      if (!(e instanceof DOMException && e.name === "AbortError")) {
        setEvents((prev) => [...prev, { type: "error", detail: String(e) }]);
      }
    } finally {
      setRunning(false);
    }
  }

  const tokens = events
    .filter((e): e is Extract<Event, { type: "token" }> => e.type === "token")
    .map((e) => e.delta)
    .join("");

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-2">
        <input
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask about pace, gaps, tyre deg…"
          className="flex-1 border-2 border-white/40 bg-transparent px-3 py-2 text-white outline-none focus:border-white"
          onKeyDown={(e) => e.key === "Enter" && ask()}
          disabled={running}
        />
        <Button onClick={ask} disabled={running}>
          {running ? "…" : "Ask"}
        </Button>
      </div>

      {events.map((e, i) =>
        e.type === "tool_call" ? (
          <div key={i} className="text-xs text-white/60">
            → tool <span className="text-[var(--f1-red)]">{e.tool}</span>(
            {JSON.stringify(e.args)})
          </div>
        ) : e.type === "tool_result" ? (
          <pre key={i} className="text-xs text-white/40 max-h-24 overflow-auto border border-white/10 p-2">
            {e.summary}
          </pre>
        ) : e.type === "error" ? (
          <div key={i} className="text-sm text-red-400">
            error: {e.detail}
          </div>
        ) : null,
      )}

      {tokens ? (
        <div className="whitespace-pre-wrap text-white leading-relaxed">{tokens}</div>
      ) : null}
    </div>
  );
}

function parseSSE(block: string): Event | null {
  let ev: string | null = null;
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) ev = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!ev) return null;
  const data = dataLines.join("\n");
  try {
    const parsed = JSON.parse(data);
    switch (ev) {
      case "token":
        return { type: "token", delta: parsed.delta ?? "" };
      case "tool_call":
        return { type: "tool_call", tool: parsed.tool, args: parsed.args ?? {} };
      case "tool_result":
        return { type: "tool_result", tool: parsed.tool, summary: parsed.summary ?? "" };
      case "done":
        return { type: "done", ...parsed };
      case "error":
        return { type: "error", detail: parsed.detail ?? parsed.type ?? "unknown" };
      default:
        return null;
    }
  } catch {
    return null;
  }
}
