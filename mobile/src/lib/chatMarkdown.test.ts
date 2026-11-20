import { describe, expect, it } from 'vitest'
import { renderChatMarkdown } from './chatMarkdown'

describe('assistant Markdown', () => {
  it('formats model-spaced bold, nested lists and italic notes', () => {
    const source = [
      '1. **四类体系（火星项目采用） **',
      '   SL/SP/FL/CL',
      '   - SL：滑移型',
      '   - SP：滑塌型',
      '2. **六类体系（跨天体比较适用）**',
      '   - 流动型',
      '',
      '*说明：请以编目为准。*',
    ].join('\n')

    const html = renderChatMarkdown(source)
    expect(html).toContain('<ol>')
    expect(html).toContain('<li><strong>四类体系（火星项目采用）</strong><br>')
    expect(html).toContain('<ul>\n<li>SL：滑移型</li>\n<li>SP：滑塌型</li>')
    expect(html).toContain('<strong>六类体系（跨天体比较适用）</strong>')
    expect(html).toContain('<em>说明：请以编目为准。</em>')
  })

  it('leaves code examples and incomplete streaming markers alone', () => {
    expect(renderChatMarkdown('**四类体系')).toContain('**四类体系')
    expect(renderChatMarkdown('**四类体系 **')).toContain('<strong>四类体系</strong>')
    expect(renderChatMarkdown('`**原样 **`')).toContain('<code>**原样 **</code>')
    expect(renderChatMarkdown('```text\n**原样 **\n```')).toContain('**原样 **')
  })

  it('escapes raw elements and only activates web destinations', () => {
    const html = renderChatMarkdown(
      '<details>文字</details>\n\n' +
        '[论文](https://example.org/paper) [文件](ftp://example.org/file) ' +
        '![原图](https://example.org/photo.png)',
    )
    expect(html).toContain('&lt;details&gt;')
    expect(html).toContain(
      '<a href="https://example.org/paper" target="_blank" rel="noopener noreferrer">论文</a>',
    )
    expect(html).not.toContain('href="ftp:')
    expect(html).not.toContain('<img')
  })
})
