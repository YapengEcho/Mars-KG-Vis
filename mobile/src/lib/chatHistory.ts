/** Message shape accepted by the agent endpoint. */
export interface HistoryMessage {
  role: 'user' | 'assistant'
  content: string
}

/** Hard limit enforced by the backend (`api/agent/schemas.py`). */
export const MAX_HISTORY_MESSAGES = 50

/**
 * How many trailing messages the client sends per request.
 *
 * The page keeps the whole thread for display, so it grows without bound. The
 * window stays well below the backend limit so that later turns never cross it.
 */
export const HISTORY_WINDOW = 20

export interface ThreadMessage {
  role: 'user' | 'assistant'
  content: string
  /** True while an assistant reply is still streaming, or after it was cut off. */
  pending?: boolean
}

/**
 * Build the `messages` payload for one request.
 *
 * Sending the full transcript eventually exceeds the backend limit and makes
 * every later request fail with 422. Keep only the most recent window, and drop
 * leading assistant messages so the history always starts with a user turn.
 *
 * Interrupted replies are excluded: a half-streamed bubble may end mid-sentence
 * and must not be replayed to the model as if it were a completed answer.
 */
export function buildChatHistory(
  thread: ReadonlyArray<ThreadMessage>,
  window: number = HISTORY_WINDOW,
): HistoryMessage[] {
  const usable: HistoryMessage[] = thread
    .filter((message) => message.role === 'user' || message.role === 'assistant')
    .filter((message) => !message.pending)
    .filter((message) => message.content.trim().length > 0)
    .map((message) => ({ role: message.role, content: message.content }))

  const limit = Math.max(1, Math.min(window, MAX_HISTORY_MESSAGES))
  const recent = usable.slice(-limit)

  while (recent.length > 0 && recent[0].role !== 'user') recent.shift()
  return recent
}
