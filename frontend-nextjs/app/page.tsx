"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { Cpu, SendHorizontal, TerminalSquare } from "lucide-react";

type MessageRole = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  streaming: boolean;
};

type TypewriterTextProps = {
  text: string;
  isActive: boolean;
  speed?: number;
  onComplete?: () => void;
};

function TypewriterText({
  text,
  isActive,
  speed = 24,
  onComplete,
}: TypewriterTextProps) {
  const [displayLength, setDisplayLength] = useState(() =>
    isActive ? 0 : text.length,
  );
  const completionSentRef = useRef(false);

  useEffect(() => {
    completionSentRef.current = false;

    if (!isActive) {
      return;
    }

    const timer = window.setInterval(() => {
      setDisplayLength((previousLength) => {
        const nextLength = Math.min(previousLength + 1, text.length);

        if (nextLength >= text.length) {
          window.clearInterval(timer);

          if (!completionSentRef.current) {
            completionSentRef.current = true;
            onComplete?.();
          }
        }

        return nextLength;
      });
    }, speed);

    return () => window.clearInterval(timer);
  }, [isActive, onComplete, speed, text]);

  const visibleText = isActive ? text.slice(0, displayLength) : text;

  return (
    <p className="whitespace-pre-wrap leading-relaxed text-cyan-50">
      {visibleText}
      {isActive && displayLength < text.length ? (
        <span className="animate-blink text-cyan-300">|</span>
      ) : null}
    </p>
  );
}

function StatusTag({ label, colorClass }: { label: string; colorClass: string }) {
  return (
    <div className="flex items-center gap-2 rounded-full border border-zinc-700/80 bg-zinc-900/70 px-3 py-1.5">
      <span className={clsx("h-2 w-2 rounded-full", colorClass)} />
      <span className="text-[10px] font-mono tracking-[0.2em] text-zinc-300">
        {label}
      </span>
    </div>
  );
}

function createMessage(
  role: MessageRole,
  content: string,
  streaming = false,
): ChatMessage {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    role,
    content,
    timestamp: new Date().toISOString(),
    streaming,
  };
}

function formatTimestamp(isoTimestamp: string): string {
  return new Date(isoTimestamp).toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    createMessage(
      "assistant",
      "OMNIMIND console online. Enter a prompt to route through GO-GATEWAY and begin memory distillation.",
      false,
    ),
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollContainerRef.current?.scrollTo({
      top: scrollContainerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, typingMessageId]);

  const handleTypewriterDone = useCallback((messageId: string) => {
    setMessages((previousMessages) =>
      previousMessages.map((message) =>
        message.id === messageId ? { ...message, streaming: false } : message,
      ),
    );
    setTypingMessageId((currentId) => (currentId === messageId ? null : currentId));
    setIsTyping(false);
  }, []);

  const sendMessage = useCallback(
    async (event?: React.FormEvent<HTMLFormElement>) => {
      event?.preventDefault();

      const trimmedInput = inputValue.trim();
      if (!trimmedInput || isTyping) {
        return;
      }

      setMessages((previousMessages) => [
        ...previousMessages,
        createMessage("user", trimmedInput),
      ]);
      setInputValue("");
      setIsTyping(true);

      const payload = {
        user_id: "demo_user_1",
        session_id: "session_alpha_01",
        content: trimmedInput,
      };

      try {
        const response = await fetch("http://localhost:8080/api/intake", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        let apiData: Record<string, unknown> | null = null;
        try {
          apiData = (await response.json()) as Record<string, unknown>;
        } catch {
          apiData = null;
        }

        await new Promise((resolve) => window.setTimeout(resolve, 1000));

        const gatewayMessage =
          typeof apiData?.message === "string"
            ? apiData.message
            : "Distillation queued successfully.";

        const aiResponse = response.ok
          ? `GO-GATEWAY uplink accepted the packet.\n${gatewayMessage}\n\nRunning synthetic cognition pass for: \"${trimmedInput}\"`
          : "Gateway returned a non-OK status. Check service logs and retry the request.";

        const assistantMessage = createMessage("assistant", aiResponse, true);
        setTypingMessageId(assistantMessage.id);
        setMessages((previousMessages) => [...previousMessages, assistantMessage]);
      } catch {
        await new Promise((resolve) => window.setTimeout(resolve, 1000));

        const fallbackResponse = createMessage(
          "assistant",
          "Unable to reach http://localhost:8080/api/intake. Verify docker services, then retry.",
          true,
        );

        setTypingMessageId(fallbackResponse.id);
        setMessages((previousMessages) => [...previousMessages, fallbackResponse]);
      }
    },
    [inputValue, isTyping],
  );

  return (
    <div className="relative min-h-screen overflow-hidden bg-gray-950 text-zinc-100">
      <div className="cyber-grid pointer-events-none absolute inset-0 opacity-40" />
      <div className="pointer-events-none absolute -left-16 top-8 h-72 w-72 rounded-full bg-cyan-400/20 blur-3xl" />
      <div className="pointer-events-none absolute -right-24 top-20 h-96 w-96 rounded-full bg-fuchsia-500/15 blur-3xl" />

      <main className="relative mx-auto flex min-h-screen w-full max-w-6xl flex-col px-4 py-4 sm:px-6 lg:px-8">
        <header className="glass-panel border-glow-cyan mb-4 rounded-2xl border border-cyan-300/35 px-4 py-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="rounded-lg border border-cyan-300/40 bg-cyan-400/10 p-2">
                <TerminalSquare className="h-5 w-5 text-cyan-300" />
              </div>
              <div>
                <p className="font-display text-xs tracking-[0.35em] text-cyan-200 text-glow-cyan">
                  OMNIMIND // CYBER CONSOLE
                </p>
                <p className="mt-1 font-mono text-[11px] tracking-[0.16em] text-zinc-400">
                  SESSION: session_alpha_01
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <StatusTag
                label="NEO4J: ONLINE (GRAPH SYNCED)"
                colorClass="bg-emerald-400 shadow-[0_0_14px_rgba(74,222,128,0.9)]"
              />
              <StatusTag
                label="QDRANT: ACTIVE"
                colorClass="bg-cyan-300 shadow-[0_0_14px_rgba(34,211,238,0.95)]"
              />
              <StatusTag
                label="GO-GATEWAY: CONNECTED"
                colorClass="bg-emerald-400 shadow-[0_0_14px_rgba(74,222,128,0.9)]"
              />
            </div>
          </div>

          <div className="mt-3 flex items-center gap-2">
            <Cpu className="h-4 w-4 text-cyan-300" />
            <p className="font-mono text-[11px] tracking-[0.24em] text-zinc-300">
              {isTyping ? (
                <span className="text-glow-cyan">PROCESSING...</span>
              ) : (
                <span className="text-glow-green">SYSTEM STABLE</span>
              )}
            </p>
          </div>
        </header>

        <section className="glass-panel flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-zinc-700/75">
          <div
            ref={scrollContainerRef}
            className="terminal-scroll flex-1 space-y-4 overflow-y-auto p-4 sm:p-5"
          >
            {messages.map((message) => {
              const isAssistant = message.role === "assistant";
              const isCurrentlyTyping =
                isAssistant && message.streaming && typingMessageId === message.id;

              return (
                <article
                  key={message.id}
                  className={clsx(
                    "max-w-[94%] rounded-2xl border px-4 py-3 backdrop-blur-lg sm:max-w-[80%]",
                    isAssistant
                      ? "mr-auto border-cyan-300/45 border-glow-cyan bg-cyan-500/10"
                      : "ml-auto border-fuchsia-300/45 border-glow-purple bg-fuchsia-500/10",
                  )}
                >
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <span
                      className={clsx(
                        "font-mono text-[10px] tracking-[0.24em]",
                        isAssistant ? "text-cyan-100" : "text-fuchsia-100",
                      )}
                    >
                      {isAssistant ? "OMNIMIND AI" : "OPERATIVE"}
                    </span>
                    <span className="font-mono text-[10px] tracking-[0.2em] text-zinc-400">
                      {formatTimestamp(message.timestamp)}
                    </span>
                  </div>

                  {isAssistant ? (
                    <TypewriterText
                      text={message.content}
                      isActive={isCurrentlyTyping}
                      speed={24}
                      onComplete={() => handleTypewriterDone(message.id)}
                    />
                  ) : (
                    <p className="whitespace-pre-wrap leading-relaxed text-fuchsia-50">
                      {message.content}
                    </p>
                  )}
                </article>
              );
            })}
          </div>

          <form
            onSubmit={sendMessage}
            className="sticky bottom-0 border-t border-zinc-800/80 bg-gray-950/85 p-4 backdrop-blur-xl sm:p-5"
          >
            <div className="border-glow-cyan flex items-center gap-3 rounded-xl border border-cyan-300/35 bg-zinc-950/90 px-3 py-2">
              <input
                value={inputValue}
                onChange={(event) => setInputValue(event.target.value)}
                placeholder="Inject command into omnichannel memory pipeline..."
                className="w-full bg-transparent font-mono text-sm text-zinc-100 outline-none placeholder:text-zinc-500"
                disabled={isTyping}
              />
              <button
                type="submit"
                disabled={!inputValue.trim() || isTyping}
                className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-cyan-300/45 bg-cyan-400/15 text-cyan-200 transition hover:bg-cyan-400/25 disabled:cursor-not-allowed disabled:opacity-45"
                aria-label="Send message"
              >
                <SendHorizontal className="h-4 w-4" />
              </button>
            </div>
          </form>
        </section>
      </main>
    </div>
  );
}
