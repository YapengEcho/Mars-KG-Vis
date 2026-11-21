/** Largest zoom applied when focusing a single detection candidate. */
const MAX_CANDIDATE_ZOOM = 8

export interface BoxTransform {
  scale: number
  x: number
  y: number
}

/**
 * Transform that centres one candidate box inside the preview stage.
 * `viewportWidth` is the rendered width of the whole image; the stage keeps the
 * image aspect ratio, and boxes use the task input's pixel coordinates.
 */
export function fitCandidateBox(
  box: readonly [number, number, number, number],
  image: { width: number; height: number },
  viewportWidth: number,
): BoxTransform {
  const [x0, y0, x1, y1] = box
  const boxWidth = x1 - x0
  const boxHeight = y1 - y0
  if (
    !(viewportWidth > 0) ||
    !(image.width > 0) ||
    !(image.height > 0) ||
    !(boxWidth > 0) ||
    !(boxHeight > 0)
  )
    return { scale: 1, x: 0, y: 0 }
  const stageHeight = (viewportWidth * image.height) / image.width
  const renderedWidth = (boxWidth / image.width) * viewportWidth
  const renderedHeight = (boxHeight / image.height) * stageHeight
  const scale = Math.min(
    MAX_CANDIDATE_ZOOM,
    Math.max(1, Math.min(viewportWidth / renderedWidth, stageHeight / renderedHeight)),
  )
  const centreX = (((x0 + x1) / 2) / image.width) * viewportWidth
  const centreY = (((y0 + y1) / 2) / image.height) * stageHeight
  return {
    scale,
    x: viewportWidth / 2 - centreX * scale,
    y: stageHeight / 2 - centreY * scale,
  }
}
