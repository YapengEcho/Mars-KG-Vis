/** Yield data from complete server-sent events, including CRLF-delimited streams. */
export async function* chatEventData(body: ReadableStream<Uint8Array>) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let ended = false
  try {
    while (!ended) {
      const { value, done } = await reader.read()
      ended = done
      buffer += decoder.decode(value, { stream: !done })
      let boundary = /\r\n\r\n|\r\n\n|\n\r\n|\n\n|\r\r/.exec(buffer)
      while (boundary) {
        const event = buffer.slice(0, boundary.index)
        buffer = buffer.slice(boundary.index + boundary[0].length)
        const data = event
          .split(/\r\n?|\n/)
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).replace(/^ /, ''))
          .join('\n')
        if (data) yield data
        boundary = /\r\n\r\n|\r\n\n|\n\r\n|\n\n|\r\r/.exec(buffer)
      }
    }
  } finally {
    if (!ended) await reader.cancel().catch(() => {})
    reader.releaseLock()
  }
}
