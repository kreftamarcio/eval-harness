/**
 * eval-harness: measure whether a change to an LLM system actually helped.
 *
 * Four layers:
 *   EvalRunner     -> execute a suite against a model, with concurrency and retries
 *   scorers        -> turn a response into a number
 *   Bootstrap      -> is the difference real, or is it noise
 *   ParetoAnalyzer -> is the better model worth what it costs
 */

import { EvalRunner } from './core/runner.js';
import { Bootstrap } from './statistics/bootstrap.js';
import { ParetoAnalyzer } from './statistics/pareto.js';

import type {
  EvalSuite,
  EvalCase,
  ScoringConfig,
  Scorer,
  ScoreResult,
  ModelConfig,
  EvalResult,
  CaseResult,
  AggregateMetrics,
  RunnerConfig,
} from './core/runner.js';
import type {
  ConfidenceInterval,
  PairedComparison,
  BootstrapConfig,
} from './statistics/bootstrap.js';
import type {
  EvaluationPoint,
  DominanceResult,
  FrontierSegment,
  ParetoAnalysis,
  Objective,
} from './statistics/pareto.js';

export { EvalRunner, Bootstrap, ParetoAnalyzer };

export {
  exactMatch,
  containsMatch,
  semanticSimilarity,
  llmJudge,
  formatCompliance,
  lengthCheck,
} from './scorers/builtin.js';

export type {
  EvalSuite,
  EvalCase,
  ScoringConfig,
  Scorer,
  ScoreResult,
  ModelConfig,
  EvalResult,
  CaseResult,
  AggregateMetrics,
  RunnerConfig,
  ConfidenceInterval,
  PairedComparison,
  BootstrapConfig,
  EvaluationPoint,
  DominanceResult,
  FrontierSegment,
  ParetoAnalysis,
  Objective,
};

export interface AlignedComparison extends PairedComparison {
  baselineModel: string;
  candidateModel: string;
  /** Case ids present in both runs and compared. */
  comparedCaseIds: string[];
  /** Case ids dropped because one run had no usable score for them. */
  skippedCaseIds: string[];
  /** Human-readable verdict, safe to print in CI output. */
  summary: string;
}

export interface CompareRunsOptions {
  bootstrap?: Partial<BootstrapConfig>;
  /**
   * Treat errored cases as a score of 0 instead of dropping them.
   *
   * Dropping is the default because a run that crashed on its ten hardest
   * cases would otherwise look like an improvement. Scoring them as 0 is the
   * right call only when a failure genuinely means "wrong answer" for you.
   */
  countErrorsAsZero?: boolean;
}

/**
 * Compare two runs of the same suite.
 *
 * Pairs by case id rather than array position. Position-based pairing looks
 * correct until one run retries in a different order or drops a case, at which
 * point it silently compares unrelated examples and reports a confident number
 * about nothing.
 */
export function compareRuns(
  baseline: EvalResult,
  candidate: EvalResult,
  options: CompareRunsOptions = {},
): AlignedComparison {
  const usable = (result: CaseResult): boolean =>
    options.countErrorsAsZero ? true : result.error === undefined;

  const scoreOf = (result: CaseResult): number =>
    result.error !== undefined && options.countErrorsAsZero ? 0 : result.aggregateScore;

  const baselineById = new Map(baseline.cases.map(c => [c.caseId, c]));
  const candidateById = new Map(candidate.cases.map(c => [c.caseId, c]));

  const comparedCaseIds: string[] = [];
  const skippedCaseIds: string[] = [];
  const baseScores: number[] = [];
  const candScores: number[] = [];

  // Iterate the baseline in its original order so output is deterministic.
  for (const baseCase of baseline.cases) {
    const candCase = candidateById.get(baseCase.caseId);

    if (!candCase || !usable(baseCase) || !usable(candCase)) {
      skippedCaseIds.push(baseCase.caseId);
      continue;
    }

    comparedCaseIds.push(baseCase.caseId);
    baseScores.push(scoreOf(baseCase));
    candScores.push(scoreOf(candCase));
  }

  // Cases the candidate ran that the baseline never covered cannot be paired.
  for (const candCase of candidate.cases) {
    if (!baselineById.has(candCase.caseId)) {
      skippedCaseIds.push(candCase.caseId);
    }
  }

  if (comparedCaseIds.length === 0) {
    throw new Error(
      'No comparable cases between the two runs. They share no case ids, or ' +
      'every shared case errored in at least one run.',
    );
  }

  const comparison = new Bootstrap(options.bootstrap).comparePaired(
    baseScores,
    candScores,
  );

  const delta = comparison.meanDifference;
  const { lower, upper } = comparison.confidenceInterval;

  const direction = delta > 0 ? 'better' : 'worse';
  const summary = comparison.significant
    ? `${candidate.modelName} is ${direction} than ${baseline.modelName}: ` +
      `${(delta * 100).toFixed(1)}% ` +
      `[${(lower * 100).toFixed(1)}%, ${(upper * 100).toFixed(1)}%], ` +
      `effect ${comparison.effectSizeLabel}, n=${comparison.sampleSize}`
    : `No significant difference between ${candidate.modelName} and ` +
      `${baseline.modelName}: ${(delta * 100).toFixed(1)}% ` +
      `[${(lower * 100).toFixed(1)}%, ${(upper * 100).toFixed(1)}%] ` +
      `includes zero, n=${comparison.sampleSize}`;

  return {
    ...comparison,
    baselineModel: baseline.modelName,
    candidateModel: candidate.modelName,
    comparedCaseIds,
    skippedCaseIds,
    summary,
  };
}

/**
 * Turn a set of runs into Pareto points.
 *
 * Cost is supplied per model rather than derived, because the harness does not
 * observe token usage or pricing. Guessing it would be the least defensible
 * number in the whole report.
 */
export function toParetoPoints(
  results: EvalResult[],
  costPerThousandRequests: Record<string, number>,
): EvaluationPoint[] {
  return results.map(result => {
    const cost = costPerThousandRequests[result.modelName];

    if (cost === undefined) {
      throw new Error(
        `No cost provided for model "${result.modelName}". ` +
        'Pareto analysis compares quality against cost, so every model needs one.',
      );
    }

    const latencies = result.cases
      .filter(c => c.error === undefined)
      .map(c => c.latencyMs)
      .sort((a, b) => a - b);

    return {
      id: result.modelName,
      quality: result.aggregate.mean,
      cost,
      latency: latencies.length > 0 ? percentile(latencies, 0.95) : undefined,
      metadata: {
        suiteId: result.suiteId,
        caseCount: result.cases.length,
        errorCount: result.cases.filter(c => c.error !== undefined).length,
        ci95: result.aggregate.ci95,
      },
    };
  });
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;

  const index = p * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (lower === upper) return sorted[lower]!;

  const weight = index - lower;
  return sorted[lower]! * (1 - weight) + sorted[upper]! * weight;
}
