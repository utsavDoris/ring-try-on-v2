type Point = { x: number; y: number };
export type PixelImage = { width: number; height: number; data: ArrayLike<number> };
// A conservative image-quality score, not a probability of correct segmentation.
export const MIN_FINGER_EDGE_CONFIDENCE = 0.6;
/** Outer pair may win when its score is at least this fraction of the peak score. */
const OUTER_PAIR_SCORE_RATIO = 0.55;
export type FingerEdgeReason = 'clear' | 'weak contrast' | 'competing edges' | 'inconsistent sections' | 'boundary too close';
export type FingerSilhouette = {
  left: Point; right: Point; width: number; support: number;
  confidence: number; reason: FingerEdgeReason;
  sections: { left: Point; right: Point }[];
};
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const clamp = (value: number) => Math.max(0, Math.min(1, value));

/** Median cross-section width in the same space as section endpoints. */
export function sectionMedianWidth(silhouette: FingerSilhouette): number {
  if (!silhouette.sections.length) return silhouette.width;
  const widths = silhouette.sections.map((section) =>
    Math.hypot(section.right.x - section.left.x, section.right.y - section.left.y),
  );
  return median(widths);
}

/**
 * Soft along-bone seat from silhouette taper.
 * Palm-thicker fingers nudge toward MCP (lower t); tip-thicker toward PIP.
 */
export function softSeatCrossPosition(
  silhouette: FingerSilhouette | null | undefined,
  base = 0.38,
  min = 0.32,
  max = 0.44,
): number {
  if (!silhouette?.sections || silhouette.sections.length < 4) {
    return Math.max(min, Math.min(max, base));
  }
  const widths = silhouette.sections.map((section) =>
    Math.hypot(section.right.x - section.left.x, section.right.y - section.left.y),
  );
  const mid = Math.floor(widths.length / 2);
  const palm = median(widths.slice(0, mid));
  const tip = median(widths.slice(mid));
  if (!(palm > 0 && tip > 0)) return Math.max(min, Math.min(max, base));
  // Positive taper (palm wider) → seat more palm-ward (smaller t).
  const taper = (palm - tip) / palm;
  const nudge = Math.max(-0.06, Math.min(0.06, taper * 0.08));
  return Math.max(min, Math.min(max, base - nudge));
}

type EdgePeak = { at: number; strength: number };
type EdgePair = { left: number; right: number; score: number; contrast: number };

/**
 * Among valid left/right pairs, prefer a modestly wider outer span when its
 * contrast is competitive — so ring diameter matches skin edge-to-edge, not an
 * internal highlight. Do not jump to far background edges.
 */
function pickOuterEdgePair(pairs: EdgePair[]): EdgePair | null {
  if (!pairs.length) return null;
  const ranked = [...pairs].sort((a, b) => b.score - a.score);
  const best = ranked[0];
  const bestWidth = best.right - best.left;
  const maxOut = Math.max(8, bestWidth * 0.18);
  const outerCompetitive = ranked.filter((pair) => {
    if (pair.score < best.score * OUTER_PAIR_SCORE_RATIO) return false;
    // Must be the same span or an outward expansion of the strongest pair.
    if (pair.left > best.left + 1 || pair.right < best.right - 1) return false;
    if (best.left - pair.left > maxOut || pair.right - best.right > maxOut) return false;
    return true;
  });
  return outerCompetitive.sort((a, b) => (b.right - b.left) - (a.right - a.left))[0] ?? best;
}

// Sample several cross-sections on either side of the attachment. Measuring
// one row made a palm crease, shadow, or one noisy pixel control ring size.
export function findFingerSilhouette(
  image: PixelImage, base: Point, tip: Point,
  neighbors: { base: Point; tip: Point }[] = [], position = 0.53,
): FingerSilhouette | null {
  const dx = tip.x - base.x, dy = tip.y - base.y;
  const length = Math.hypot(dx, dy);
  if (length < 8) return null;
  const ux = dx / length, uy = dy / length;
  const nx = -uy, ny = ux;
  const origin = { x: base.x + dx * position, y: base.y + dy * position };
  const sample = (x: number, y: number, channel: number) => {
    const ix = Math.round(x), iy = Math.round(y);
    if (ix < 1 || iy < 1 || ix >= image.width - 1 || iy >= image.height - 1) return NaN;
    return image.data[(iy * image.width + ix) * 4 + channel];
  };
  const sections: {
    left: number; right: number; cx: number; cy: number;
    contrast: number; ambiguity: number; nearBoundary: boolean;
  }[] = [];
  const stationScale = length < 24 ? Math.max(0.6, length / 24) : 1;
  const stations = [-0.18, -0.12, -0.06, 0.06, 0.12, 0.18].map(s => s * stationScale);
  for (const station of stations) {
    const cx = origin.x + dx * station, cy = origin.y + dy * station;
    let lower = -length * 0.62, upper = length * 0.62;
    for (const neighbor of neighbors) {
      const t = position + station;
      const qx = neighbor.base.x + (neighbor.tip.x - neighbor.base.x) * t;
      const qy = neighbor.base.y + (neighbor.tip.y - neighbor.base.y) * t;
      const across = (qx - cx) * nx + (qy - cy) * ny;
      // Clip scans short of the neighbor centerline (keep headroom for our own edge).
      if (across < 0) lower = Math.max(lower, across * 0.60);
      else upper = Math.min(upper, across * 0.60);
    }
    const candidates = (start: number, end: number): EdgePeak[] => {
      const values: { at: number; strength: number; direction: number }[] = [];
      // Far hands (short bones) need a lower contrast floor; near hands stay strict.
      const minStrength = length < 16 ? 8 : length < 28 ? 10 : 14;
      for (let at = Math.ceil(start); at <= Math.floor(end); at++) {
        let strength = 0, direction = 0;
        for (let c = 0; c < 3; c++) {
          // Average along the finger, compare both sides of the boundary.
          let before = 0, after = 0;
          for (let row = -1; row <= 1; row++) {
            before += sample(cx + nx * (at - 2) + ux * row, cy + ny * (at - 2) + uy * row, c);
            after += sample(cx + nx * (at + 2) + ux * row, cy + ny * (at + 2) + uy * row, c);
          }
          strength += Math.abs(after - before) / 9;
          direction += (after - before) / 9;
        }
        if (Number.isFinite(strength) && strength > minStrength) values.push({ at, strength, direction });
      }
      // Merge a gradient plateau into one boundary. Otherwise one physical
      // edge produces several competing peaks and a pixel-dependent position.
      const peaks: EdgePeak[] = [];
      for (let i = 0; i < values.length;) {
        let weighted = 0, weight = 0, strength = 0;
        let j = i;
        do {
          weighted += values[j].at * values[j].strength;
          weight += values[j].strength;
          strength = Math.max(strength, values[j].strength);
          j++;
        } while (j < values.length && values[j].at - values[j - 1].at <= 1
          && values[j].direction * values[j - 1].direction >= 0);
        peaks.push({ at: weighted / weight, strength });
        i = j;
      }
      return peaks.sort((a, b) => b.strength - a.strength).slice(0, 6);
    };
    const left = candidates(lower, -length * 0.13);
    const right = candidates(length * 0.13, upper);
    // Far-hand pixel quantization: slightly wider anatomical width band.
    const minWidthRatio = length < 20 ? 0.25 : 0.3;
    const maxWidthRatio = length < 20 ? 0.92 : 0.85;
    const pairs: EdgePair[] = [];
    for (const a of left) for (const b of right) {
      const width = b.at - a.at, offset = (a.at + b.at) / 2;
      // Reject dual-finger spans and midpoints far from the bone axis.
      // Hard finger-width clamp vs bone length happens later in resolveFingerWidthPx.
      if (width < length * minWidthRatio || width > length * maxWidthRatio || Math.abs(offset) > width * 0.18) continue;
      const score = Math.sqrt(a.strength * b.strength) / (1 + Math.abs(offset) / width);
      pairs.push({ left: a.at, right: b.at, score, contrast: Math.min(a.strength, b.strength) });
    }
    const best = pickOuterEdgePair(pairs);
    if (best) {
      const separation = Math.max(2, (best.right - best.left) * 0.08);
      const alternative = pairs.find(p => Math.abs(p.left - best.left) > separation || Math.abs(p.right - best.right) > separation);
      // A strong internal highlight can be consistent along the whole finger.
      // Additional outward boundaries are evidence against trusting it, even
      // when those outer boundaries have lower contrast than the highlight.
      const outwardLeft = left.some(p => p.at < best.left - separation);
      const outwardRight = right.some(p => p.at > best.right + separation);
      const inwardLeft = left.some(p => p.at > best.left + separation);
      const inwardRight = right.some(p => p.at < best.right - separation);
      const bilateralAlternative = pairs.some(p => p.score > best.score * 0.75
        && Math.abs(p.left - best.left) > separation && Math.abs(p.right - best.right) > separation);
      // A background/shadow transition on one side is common with spread
      // fingers. Penalize it, but reserve the veto for ambiguous pairs on
      // both sides (for example a bright strip inside the finger).
      // Competing inward peaks (highlight) or outward peaks (bg) both mean the
      // span is not a single clear skin boundary.
      const ambiguity = outwardLeft && outwardRight || inwardLeft && inwardRight || bilateralAlternative ? 0.35
        : outwardLeft || outwardRight || inwardLeft || inwardRight
          || !!alternative && alternative.score > best.score * 0.75 ? 0.8 : 1;
      const nearImageEdge = (at: number) => {
        const x = cx + nx * at, y = cy + ny * at;
        return x < 4 || y < 4 || x > image.width - 5 || y > image.height - 5;
      };
      sections.push({ ...best, cx, cy,
        ambiguity,
        nearBoundary: best.left - lower < 2 || upper - best.right < 2
          || nearImageEdge(best.left) || nearImageEdge(best.right),
      });
    }

  }
  if (!sections.length) return null;
  // Webbing or a nearby neighbor can clip an individual scan. Fit from
  // the remaining sections when at least four have complete boundaries.
  const usable = sections.filter(s => !s.nearBoundary);
  const measured = usable.length >= 4 ? usable : sections;
  const middleWidth = median(measured.map(s => s.right - s.left));
  const middleOffset = median(measured.map(s => (s.left + s.right) / 2));
  const consistent = measured.filter(s => Math.abs((s.right - s.left) / middleWidth - 1) < 0.16 && Math.abs((s.left + s.right) / 2 - middleOffset) < middleWidth * 0.12);
  if (!consistent.length) return null;
  const left = median(consistent.map(s => s.left)), right = median(consistent.map(s => s.right));
  const spread = median(measured.map(s => Math.abs(s.right - s.left - middleWidth))) / middleWidth;
  const checks: { confidence: number; reason: FingerEdgeReason }[] = [
    { confidence: clamp(median(consistent.map(s => s.contrast)) / 35), reason: 'weak contrast' },
    { confidence: Math.min(consistent.length < 4 ? 0.4 : 1, consistent.length / measured.length, clamp(1 - spread / 0.2)), reason: 'inconsistent sections' },
    { confidence: [...consistent].sort((a, b) => a.ambiguity - b.ambiguity)[Math.floor((consistent.length - 1) / 2)].ambiguity, reason: 'competing edges' },
    { confidence: consistent.some(s => s.nearBoundary) ? 0.4 : 1, reason: 'boundary too close' },
  ];
  const weakest = checks.reduce((a, b) => a.confidence <= b.confidence ? a : b);
  const sectionWidths = consistent.map((s) => s.right - s.left);
  // Prefer section-median width so one highlight pair cannot set ring diameter.
  const fitWidth = sectionWidths.length ? median(sectionWidths) : right - left;
  return {
    left: { x: origin.x + nx * left, y: origin.y + ny * left },
    right: { x: origin.x + nx * right, y: origin.y + ny * right },
    width: fitWidth,
    support: consistent.length,
    confidence: weakest.confidence,
    reason: weakest.confidence >= MIN_FINGER_EDGE_CONFIDENCE ? 'clear' : weakest.reason,
    sections: sections.map(s => ({
      left: { x: s.cx + nx * s.left, y: s.cy + ny * s.left },
      right: { x: s.cx + nx * s.right, y: s.cy + ny * s.right },
    })),
  };
}
