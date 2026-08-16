import { describe, it, expect } from 'vitest';
import { Bootstrap } from '../src/statistics/bootstrap.js';
import { ParetoAnalyzer } from '../src/statistics/pareto.js';
import type { EvaluationPoint } from '../src/statistics/pareto.js';

// Fixed seed: an evaluation harness whose own numbers move between runs is
// not a harness.
const SEED = 42;

describe('Bootstrap configuration', () => {
  it('refuses too few iterations to keep intervals stable', () => {
    expect(() => new Bootstrap({ iterations: 500 })).toThrow(/below 1000/);
  });

  it('rejects a confidence level outside (0,1)', () => {
    expect(() => new Bootstrap({ confidenceLevel: 1 })).toThrow(
      /strictly between 0 and 1/,
    );
    expect(() => new Bootstrap({ confidenceLevel: 0 })).toThrow(
      /strictly between 0 and 1/,
    );
  });
});

describe('meanConfidenceInterval', () => {
  const boot = new Bootstrap({ iterations: 2000, seed: SEED });

  it('throws on an empty sample', () => {
    expect(() => boot.meanConfidenceInterval([])).toThrow(/zero samples/);
  });

  it('brackets the sample mean', () => {
    const scores = [0.6, 0.7, 0.8, 0.75, 0.65, 0.9, 0.55, 0.85];
    const ci = boot.meanConfidenceInterval(scores);

    const mean = scores.reduce((s, v) => s + v, 0) / scores.length;

    expect(ci.pointEstimate).toBeCloseTo(mean, 10);
    expect(ci.lower).toBeLessThanOrEqual(ci.pointEstimate);
    expect(ci.upper).toBeGreaterThanOrEqual(ci.pointEstimate);
    expect(ci.level).toBe(0.95);
  });

  it('collapses to a point when every score is identical', () => {
    const ci = boot.meanConfidenceInterval([0.5, 0.5, 0.5, 0.5]);

    expect(ci.lower).toBeCloseTo(0.5, 10);
    expect(ci.upper).toBeCloseTo(0.5, 10);
  });

  it('produces a narrower interval as the sample grows', () => {
    const small = boot.meanConfidenceInterval([0.2, 0.9, 0.4, 0.8, 0.3]);
    const large = boot.meanConfidenceInterval(
      Array.from({ length: 200 }, (_, i) => (i % 2 === 0 ? 0.4 : 0.6)),
    );

    expect(large.upper - large.lower).toBeLessThan(small.upper - small.lower);
  });

  it('is reproducible for a given seed', () => {
    const a = new Bootstrap({ iterations: 2000, seed: SEED });
    const b = new Bootstrap({ iterations: 2000, seed: SEED });
    const scores = [0.1, 0.5, 0.9, 0.4, 0.7];

    expect(a.meanConfidenceInterval(scores)).toEqual(
      b.meanConfidenceInterval(scores),
    );
  });
});

describe('comparePaired', () => {
  const boot = new Bootstrap({ iterations: 2000, seed: SEED });

  it('rejects arrays of different length', () => {
    expect(() => boot.comparePaired([0.5, 0.6], [0.5])).toThrow(
      /equal-length arrays/,
    );
  });

  it('rejects an empty comparison', () => {
    expect(() => boot.comparePaired([], [])).toThrow(/zero samples/);
  });

  it('detects a consistent improvement as significant', () => {
    const baseline = Array.from({ length: 40 }, (_, i) => 0.4 + (i % 5) * 0.02);
    const candidate = baseline.map(v => v + 0.15);

    const result = boot.comparePaired(baseline, candidate);

    expect(result.meanDifference).toBeCloseTo(0.15, 6);
    expect(result.significant).toBe(true);
    expect(result.confidenceInterval.lower).toBeGreaterThan(0);
  });

  it('does not claim significance for pure noise', () => {
    const baseline = [0.5, 0.6, 0.4, 0.55, 0.45, 0.5, 0.62, 0.38];
    const candidate = [0.52, 0.58, 0.42, 0.53, 0.47, 0.49, 0.6, 0.4];

    const result = boot.comparePaired(baseline, candidate);

    expect(result.significant).toBe(false);
    expect(result.confidenceInterval.lower).toBeLessThanOrEqual(0);
    expect(result.confidenceInterval.upper).toBeGreaterThanOrEqual(0);
  });

  it('reports a negative difference for a regression', () => {
    const baseline = Array.from({ length: 30 }, () => 0.8);
    const candidate = Array.from({ length: 30 }, () => 0.6);

    const result = boot.comparePaired(baseline, candidate);

    expect(result.meanDifference).toBeCloseTo(-0.2, 6);
    expect(result.confidenceInterval.upper).toBeLessThan(0);
  });

  it('labels effect size by magnitude', () => {
    const baseline = Array.from({ length: 30 }, (_, i) => 0.5 + (i % 3) * 0.01);
    const large = baseline.map(v => v + 0.3);

    expect(boot.comparePaired(baseline, large).effectSizeLabel).toBe('large');
  });

  it('reports zero effect size when nothing changed', () => {
    const scores = [0.5, 0.7, 0.3, 0.9];
    const result = boot.comparePaired(scores, scores);

    expect(result.meanDifference).toBe(0);
    expect(result.effectSize).toBe(0);
    expect(result.effectSizeLabel).toBe('negligible');
    expect(result.significant).toBe(false);
  });
});

describe('ParetoAnalyzer', () => {
  const analyzer = new ParetoAnalyzer();

  const points: EvaluationPoint[] = [
    { id: 'small', quality: 0.62, cost: 0.15 },
    { id: 'medium', quality: 0.78, cost: 0.60 },
    { id: 'large', quality: 0.84, cost: 3.00 },
    // Strictly worse than medium on both axes: must be eliminated.
    { id: 'legacy', quality: 0.70, cost: 1.20 },
  ];

  it('throws on an empty input', () => {
    expect(() => analyzer.analyze([])).toThrow(/zero points/);
  });

  it('rejects duplicate ids, since dominance is computed by id', () => {
    expect(() =>
      analyzer.analyze([
        { id: 'a', quality: 0.5, cost: 1 },
        { id: 'a', quality: 0.6, cost: 2 },
      ]),
    ).toThrow(/Duplicate configuration id/);
  });

  it('eliminates a configuration that is worse on every axis', () => {
    const analysis = analyzer.analyze(points);

    expect(analysis.dominated.map(d => d.id)).toContain('legacy');
    expect(analysis.frontier.map(p => p.id)).not.toContain('legacy');
    expect(analysis.dominated.find(d => d.id === 'legacy')!.dominatedBy).toContain(
      'medium',
    );
  });

  it('keeps every genuine trade-off on the frontier', () => {
    const analysis = analyzer.analyze(points);

    expect(analysis.frontier.map(p => p.id)).toEqual(['small', 'medium', 'large']);
  });

  it('sorts the frontier along the cost axis', () => {
    const analysis = analyzer.analyze(points);
    const costs = analysis.frontier.map(p => p.cost);

    expect([...costs].sort((a, b) => a - b)).toEqual(costs);
  });

  it('identifies the cheapest and the highest quality configurations', () => {
    const analysis = analyzer.analyze(points);

    expect(analysis.cheapest.id).toBe('small');
    expect(analysis.highestQuality.id).toBe('large');
  });

  it('computes marginal quality per dollar between frontier points', () => {
    const analysis = analyzer.analyze(points);

    expect(analysis.segments).toHaveLength(2);

    const [first, second] = analysis.segments;
    expect(first.from.id).toBe('small');
    expect(first.to.id).toBe('medium');
    expect(first.qualityPerDollar).toBeGreaterThan(second.qualityPerDollar);
  });

  it('finds the knee where marginal efficiency collapses', () => {
    const analysis = analyzer.analyze(points);

    // small -> medium buys 0.16 quality for $0.45; medium -> large buys 0.06
    // for $2.40. The knee is medium.
    expect(analysis.diminishingReturnsAt?.id).toBe('medium');
  });

  it('returns no knee for a frontier with fewer than three points', () => {
    const analysis = analyzer.analyze([
      { id: 'a', quality: 0.5, cost: 1 },
      { id: 'b', quality: 0.8, cost: 2 },
    ]);

    expect(analysis.diminishingReturnsAt).toBeNull();
  });

  it('keeps both configurations when they tie on every objective', () => {
    const analysis = analyzer.analyze([
      { id: 'a', quality: 0.7, cost: 1 },
      { id: 'b', quality: 0.7, cost: 1 },
    ]);

    expect(analysis.frontier).toHaveLength(2);
    expect(analysis.dominated).toHaveLength(0);
  });

  it('does not let an unmeasured latency dominate a measured one', () => {
    const analysis = analyzer.analyze(
      [
        { id: 'measured', quality: 0.7, cost: 1, latency: 900 },
        { id: 'unmeasured', quality: 0.7, cost: 1 },
      ],
      ['quality', 'cost', 'latency'],
    );

    expect(analysis.frontier).toHaveLength(2);
  });

  it('explains the trade-off in text a reviewer can read', () => {
    const report = analyzer.explain(analyzer.analyze(points));

    expect(report).toContain('legacy');
    expect(report).toContain('Best value');
    expect(report).toContain('Diminishing returns');
  });
});
