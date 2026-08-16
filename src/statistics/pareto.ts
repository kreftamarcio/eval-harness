/**
 * Pareto frontier analysis for cost-quality trade-offs.
 *
 * A configuration is Pareto-dominated when another configuration is at least
 * as good on every objective and strictly better on at least one. Dominated
 * configurations should never be selected: something else is better or equal
 * on all axes. Removing them shrinks a confusing many-way comparison down to
 * the genuine trade-off set.
 */

export interface EvaluationPoint {
  id: string;
  /** Higher is better. Normalized to [0,1] by the aggregator. */
  quality: number;
  /** Lower is better. USD per 1000 requests. */
  cost: number;
  /** Lower is better. Milliseconds, P95. */
  latency?: number;
  metadata?: Record<string, unknown>;
}

export interface DominanceResult extends EvaluationPoint {
  dominated: boolean;
  /** Ids of the configurations that dominate this one. */
  dominatedBy: string[];
}

export interface FrontierSegment {
  from: EvaluationPoint;
  to: EvaluationPoint;
  /** Additional cost to move from `from` to `to`. */
  costDelta: number;
  /** Quality gained by that move. */
  qualityDelta: number;
  /** Quality points purchased per additional USD. Higher is a better deal. */
  qualityPerDollar: number;
}

export interface ParetoAnalysis {
  frontier: EvaluationPoint[];
  dominated: DominanceResult[];
  segments: FrontierSegment[];
  cheapest: EvaluationPoint;
  highestQuality: EvaluationPoint;
  /** The frontier point with the best quality-per-dollar ratio overall. */
  bestValue: EvaluationPoint;
  /**
   * The point after which additional spend buys disproportionately little
   * quality. Null when the frontier has fewer than three points, since a
   * knee is not meaningfully defined below that.
   */
  diminishingReturnsAt: EvaluationPoint | null;
}

export type Objective = 'quality' | 'cost' | 'latency';

export class ParetoAnalyzer {
  /**
   * Compute the Pareto frontier over quality (maximize) and cost (minimize),
   * optionally including latency (minimize) as a third objective.
   */
  analyze(
    points: EvaluationPoint[],
    objectives: Objective[] = ['quality', 'cost'],
  ): ParetoAnalysis {
    if (points.length === 0) {
      throw new Error('Cannot compute a Pareto frontier over zero points');
    }

    this.assertUniqueIds(points);

    const dominanceResults = points.map(point => {
      const dominatedBy = points
        .filter(other => other.id !== point.id && this.dominates(other, point, objectives))
        .map(other => other.id);

      return { ...point, dominated: dominatedBy.length > 0, dominatedBy };
    });

    const frontier = dominanceResults
      .filter(r => !r.dominated)
      .map(({ dominated: _d, dominatedBy: _db, ...point }) => point)
      // Sort along the cost axis so segments read left to right
      .sort((a, b) => a.cost - b.cost);

    const dominated = dominanceResults.filter(r => r.dominated);
    const segments = this.computeSegments(frontier);

    return {
      frontier,
      dominated,
      segments,
      cheapest: frontier[0]!,
      highestQuality: this.maxBy(frontier, p => p.quality),
      bestValue: this.maxBy(frontier, p => (p.cost > 0 ? p.quality / p.cost : Infinity)),
      diminishingReturnsAt: this.findKnee(frontier, segments),
    };
  }

  /**
   * Does `a` dominate `b`?
   *
   * True when `a` is at least as good as `b` on every objective and strictly
   * better on at least one. Ties on all objectives mean neither dominates,
   * which correctly keeps both on the frontier.
   */
  private dominates(a: EvaluationPoint, b: EvaluationPoint, objectives: Objective[]): boolean {
    let strictlyBetterSomewhere = false;

    for (const objective of objectives) {
      const comparison = this.compareObjective(a, b, objective);

      if (comparison === 'worse') {
        return false; // Worse on any objective disqualifies dominance
      }
      if (comparison === 'better') {
        strictlyBetterSomewhere = true;
      }
    }

    return strictlyBetterSomewhere;
  }

  private compareObjective(
    a: EvaluationPoint,
    b: EvaluationPoint,
    objective: Objective,
  ): 'better' | 'worse' | 'equal' {
    switch (objective) {
      case 'quality':
        // Maximize
        if (a.quality > b.quality) return 'better';
        if (a.quality < b.quality) return 'worse';
        return 'equal';

      case 'cost':
        // Minimize
        if (a.cost < b.cost) return 'better';
        if (a.cost > b.cost) return 'worse';
        return 'equal';

      case 'latency': {
        // Minimize. Missing latency is treated as unknown, not as zero:
        // assuming zero would let an unmeasured config dominate a measured one.
        if (a.latency === undefined || b.latency === undefined) return 'equal';
        if (a.latency < b.latency) return 'better';
        if (a.latency > b.latency) return 'worse';
        return 'equal';
      }
    }
  }

  /**
   * Marginal analysis between consecutive frontier points.
   * This is what actually answers "is the expensive model worth it?".
   */
  private computeSegments(frontier: EvaluationPoint[]): FrontierSegment[] {
    const segments: FrontierSegment[] = [];

    for (let i = 0; i < frontier.length - 1; i++) {
      const from = frontier[i]!;
      const to = frontier[i + 1]!;

      const costDelta = to.cost - from.cost;
      const qualityDelta = to.quality - from.quality;

      segments.push({
        from,
        to,
        costDelta,
        qualityDelta,
        qualityPerDollar: costDelta > 0 ? qualityDelta / costDelta : Infinity,
      });
    }

    return segments;
  }

  /**
   * Locate the knee of the frontier: the point after which quality-per-dollar
   * drops sharply relative to the segments before it.
   *
   * Heuristic: find the largest relative drop in marginal efficiency between
   * consecutive segments. This is intentionally simple and explainable rather
   * than a curvature-fitting method, because the frontier typically has only a
   * handful of points and fitting a curve to five points invents precision
   * that isn't there.
   */
  private findKnee(
    frontier: EvaluationPoint[],
    segments: FrontierSegment[],
  ): EvaluationPoint | null {
    if (frontier.length < 3 || segments.length < 2) {
      return null;
    }

    let largestDropRatio = 0;
    let kneeIndex = -1;

    for (let i = 0; i < segments.length - 1; i++) {
      const current = segments[i]!.qualityPerDollar;
      const next = segments[i + 1]!.qualityPerDollar;

      if (!Number.isFinite(current) || current <= 0) continue;

      const dropRatio = (current - next) / current;
      if (dropRatio > largestDropRatio) {
        largestDropRatio = dropRatio;
        kneeIndex = i;
      }
    }

    // Require a meaningful drop. Below 30% the frontier is close to linear and
    // naming a knee would be misleading.
    if (kneeIndex === -1 || largestDropRatio < 0.3) {
      return null;
    }

    return segments[kneeIndex]!.to;
  }

  /**
   * Human-readable summary of the trade-off, suitable for a report or CI log.
   */
  explain(analysis: ParetoAnalysis): string {
    const lines: string[] = [];

    lines.push(`Frontier: ${analysis.frontier.length} non-dominated configuration(s).`);

    if (analysis.dominated.length > 0) {
      lines.push(`Eliminated ${analysis.dominated.length} dominated configuration(s):`);
      for (const d of analysis.dominated) {
        lines.push(`  ${d.id} is dominated by ${d.dominatedBy.join(', ')}`);
      }
    }

    lines.push('');
    lines.push(`Cheapest:        ${analysis.cheapest.id} (quality ${analysis.cheapest.quality.toFixed(3)}, cost ${analysis.cheapest.cost.toFixed(4)})`);
    lines.push(`Highest quality: ${analysis.highestQuality.id} (quality ${analysis.highestQuality.quality.toFixed(3)}, cost ${analysis.highestQuality.cost.toFixed(4)})`);
    lines.push(`Best value:      ${analysis.bestValue.id}`);

    if (analysis.diminishingReturnsAt) {
      lines.push('');
      lines.push(`Diminishing returns begin after ${analysis.diminishingReturnsAt.id}. Spending beyond this point buys disproportionately little quality.`);
    }

    if (analysis.segments.length > 0) {
      lines.push('');
      lines.push('Marginal trade-offs along the frontier:');
      for (const s of analysis.segments) {
        const q = (s.qualityDelta * 100).toFixed(1);
        lines.push(`  ${s.from.id} -> ${s.to.id}: +${q}% quality for +$${s.costDelta.toFixed(4)}`);
      }
    }

    return lines.join('\n');
  }

  private assertUniqueIds(points: EvaluationPoint[]): void {
    const seen = new Set<string>();
    for (const point of points) {
      if (seen.has(point.id)) {
        throw new Error(
          `Duplicate configuration id "${point.id}". ` +
          `Dominance is computed by id, so duplicates make the result ambiguous.`,
        );
      }
      seen.add(point.id);
    }
  }

  private maxBy<T>(items: T[], score: (item: T) => number): T {
    let best = items[0]!;
    let bestScore = score(best);

    for (let i = 1; i < items.length; i++) {
      const candidateScore = score(items[i]!);
      if (candidateScore > bestScore) {
        bestScore = candidateScore;
        best = items[i]!;
      }
    }

    return best;
  }
}
