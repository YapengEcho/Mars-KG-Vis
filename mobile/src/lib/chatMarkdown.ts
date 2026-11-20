import MarkdownIt from 'markdown-it'

export function isChatWebLink(url: string | null | undefined): url is string {
  return typeof url === 'string' && /^https?:\/\//i.test(url)
}

const parser = new MarkdownIt({ html: false, breaks: true }).disable('image')
parser.validateLink = isChatWebLink
parser.renderer.rules.link_open = (tokens, index, options, _env, renderer) => {
  tokens[index].attrSet('target', '_blank')
  tokens[index].attrSet('rel', 'noopener noreferrer')
  return renderer.renderToken(tokens, index, options)
}

parser.core.ruler.before('inline', 'normalize_spaced_bold', ({ tokens }) => {
  for (const token of tokens) {
    if (token.type !== 'inline' || !token.content.includes('**')) continue

    // Model output sometimes leaves spaces around the title inside the bold
    // markers. Skip inline code spans as well as fenced and indented code.
    token.content = token.content.replace(
      /(`+)[^\n]*?\1|(?<![\\*])\*\*([^*\n]+)\*\*(?!\*)/g,
      (match, _backticks, title: string | undefined) =>
        title?.trim() ? `**${title.trim()}**` : match,
    )
  }
})

export function renderChatMarkdown(source: string): string {
  return parser.render(source)
}
