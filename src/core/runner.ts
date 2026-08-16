/**
 * Eval Runner: orchestrates LLM evaluation pipelines.
 *
 * Executes evaluation suites against one or more models, collecting
 * structured results with statistical significance testing.
 *
 * Pipeline:
 *   1. Load eval suite (test cases + expected outputs + scoring functions)
 *   2. Execute each case against target model(s)
 *   3. Score responses using configured judges
 *   4. Aggregate metrics with confidence intervals
 *   5. Detect regressions against baseline
 *
 * Design principles:
 *   - Deterministic: same inputs always produce comparable results
 *   - Parallelizable: cases are independent, run concurrently
 *   - Extensible: custom scorers, judges, and reporters
 */

export interface EvalSuite {
  id: string;
  name: string;
  description: string;
  cases: EvalCase[];
  scoring: ScoringConfig;
  metadata?: Record<string, unknown>;
}

export interface EvalCase {
  id: string;
  input: string;
  expectedOutput?: string;
  context?: string;
  tags?: string[];
  difficulty?: 'easy' | 'medium' | 'hard';
  metadata?: Record<string, unknown>;
}

export interface ScoringConfig {
  /** Scoring functions to apply */
  scorers: Scorer[];
  /** Aggregation method */
  aggregation: 'mean' | 'median' | 'weighted';
  /** Weights per scorer (if aggregation is 'weighted') */
  weights?: Record<string, number>;
}

export interface Scorer {
  name: string;
  description?: string;
  score: (input: string, output: string, expected?: string, context?: string) => Promise<ScoreResult>;
}

export interface ScoreResult {
  value: number; // 0-1
  reasoning?: string;
  metadata?: Record<string, unknown>;
}

export interface ModelConfig {
  name: string;
  provider: string;
  model: string;
  temperature?: number;
  maxTokens?: number;
  call: (input: string) => Promise<string>;
}

export interface EvalResult {
  suiteId: string;
  modelName: string;
  timestamp: number;
  cases: CaseResult[];
  aggregate: AggregateMetrics;
  duration: number;
}

export interface CaseResult {
  caseId: string;
  input: string;
  output: string;
  expected?: string;
  scores: Record<string, ScoreResult>;
  aggregateScore: number;
  latencyMs: number;
  error?: string;
}

export interface AggregateMetrics {
  mean: number;
  median: number;
  stddev: number;
  min: number;
  max: number;
  p5: number;
  p95: number;
  /** 95% confidence interval */
  ci95: [number, number];
  /** Per-scorer breakdown */
  perScorer: Record<string, { mean: number; stddev: number }>;
  /** Per-difficulty breakdown */
  perDifficulty?: Record<string, { mean: number; count: number }>;
}

export interface RunnerConfig {
  /** Maximum concurrent evaluations */
  concurrency?: number;
  /** Timeout per case in ms */
  caseTimeout?: number;
  /** Retry failed cases */
  maxRetries?: number;
  /** Progress callback */
  onProgress?: (completed: number, total: number, latest: CaseResult) => void;
}

export class EvalRunner {
  private readonly concurrency: number;
  private readonly caseTimeout: number;
  private readonly maxRetries: number;

  constructor(private readonly config: RunnerConfig = {}) {
    this.concurrency = config.concurrency ?? 10;
    this.caseTimeout = config.caseTimeout ?? 30_000;
    this.maxRetries = config.maxRetries ?? 2;
  }

  /**
   * Run a complete evaluation suite against a model.
   */
  async run(suite: EvalSuite, model: ModelConfig): Promise<EvalResult> {
    const startTime = performance.now();
    const results: CaseResult[] = [];
    let completed = 0;

    // Process cases with concurrency control
    const batches = this.chunk(suite.cases, this.concurrency);

    for (const batch of batches) {
      const batchResults = await Promise.all(
        batch.map(evalCase => this.evaluateCase(evalCase, model, suite.scoring)),
      );

      for (const result of batchResults) {
        results.push(result);
        completed++;
        this.config.onProgress?.(completed, suite.cases.length, result);
      }
    }

    const aggregate = this.computeAggregateMetrics(results, suite.scoring);

    return {
      suiteId: suite.id,
      modelName: model.name,
      timestamp: Date.now(),
      cases: results,
      aggregate,
      duration: performance.now() - startTime,
    };
  }

  /**
   * Compare two model results and detect significant differences.
   */
  compare(baseline: EvalResult, candidate: EvalResult): {
    improved: boolean;
    delta: number;
    significant: boolean;
    pValue: number;
    summary: string;
  } {
    const baseScores = baseline.cases.map(c => c.aggregateScore);
    const candScores = candidate.cases.map(c => c.aggregateScore);

    const delta = candidate.aggregate.mean - baseline.aggregate.mean;
    const { significant, pValue } = this.pairedTTest(baseScores, candScores);

    const summary = significant
      ? `${candidate.modelName} is ${delta > 0 ? 'better' : 'worse'} than ${baseline.modelName} (p=${pValue.toFixed(4)}, delta=${(delta * 100).toFixed(1)}%)`
      : `No significant difference between ${candidate.modelName} and ${baseline.modelName} (p=${pValue.toFixed(4)})`;

    return { improved: delta > 0 && significant, delta, significant, pValue, summary };
  }

  private async evaluateCase(
    evalCase: EvalCase,
    model: ModelConfig,
    scoring: ScoringConfig,
  ): Promise<CaseResult> {
    let lastError: string | undefined;

    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const startTime = performance.now();

        // Call the model
        const output = await Promise.race([
          model.call(evalCase.input),
          this.timeout(this.caseTimeout),
        ]) as string;

        const latencyMs = performance.now() - startTime;

        // Score the output
        const scores: Record<string, ScoreResult> = {};
        for (const scorer of scoring.scorers) {
          scores[scorer.name] = await scorer.score(
            evalCase.input,
            output,
            evalCase.expectedOutput,
            evalCase.context,
          );
        }

        // Aggregate scores
        const aggregateScore = this.aggregateScores(scores, scoring);

        return {
          caseId: evalCase.id,
          input: evalCase.input,
          output,
          expected: evalCase.expectedOutput,
          scores,
          aggregateScore,
          latencyMs,
        };
      } catch (error) {
        lastError = (error as Error).message;
      }
    }

    // All retries exhausted
    return {
      caseId: evalCase.id,
      input: evalCase.input,
      output: '',
      expected: evalCase.expectedOutput,
      scores: {},
      aggregateScore: 0,
      latencyMs: 0,
      error: lastError,
    };
  }

  private aggregateScores(scores: Record<string, ScoreResult>, config: ScoringConfig): number {
    const values = Object.entries(scores).map(([name, result]) => {
      const weight = config.weights?.[name] ?? 1;
      return { value: result.value, weight };
    });

    if (values.length === 0) return 0;

    switch (config.aggregation) {
      case 'mean':
        return values.reduce((sum, v) => sum + v.value, 0) / values.length;
      case 'weighted': {
        const totalWeight = values.reduce((sum, v) => sum + v.weight, 0);
        return values.reduce((sum, v) => sum + v.value * v.weight, 0) / totalWeight;
      }
      case 'median': {
        const sorted = values.map(v => v.value).sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!;
      }
      default:
        return values.reduce((sum, v) => sum + v.value, 0) / values.length;
    }
  }

  private computeAggregateMetrics(results: CaseResult[], scoring: ScoringConfig): AggregateMetrics {
    const scores = results.map(r => r.aggregateScore);
    const n = scores.length;

    if (n === 0) {
      return { mean: 0, median: 0, stddev: 0, min: 0, max: 0, p5: 0, p95: 0, ci95: [0, 0], perScorer: {} };
    }

    const sorted = [...scores].sort((a, b) => a - b);
    const mean = scores.reduce((s, v) => s + v, 0) / n;
    const variance = scores.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1 || 1);
    const stddev = Math.sqrt(variance);

    const ci95Margin = 1.96 * (stddev / Math.sqrt(n));

    // Per-scorer metrics
    const perScorer: Record<string, { mean: number; stddev: number }> = {};
    for (const scorer of scoring.scorers) {
      const scorerValues = results
        .map(r => r.scores[scorer.name]?.value)
        .filter((v): v is number => v !== undefined);

      if (scorerValues.length > 0) {
        const sMean = scorerValues.reduce((s, v) => s + v, 0) / scorerValues.length;
        const sVar = scorerValues.reduce((s, v) => s + (v - sMean) ** 2, 0) / (scorerValues.length - 1 || 1);
        perScorer[scorer.name] = { mean: sMean, stddev: Math.sqrt(sVar) };
      }
    }

    return {
      mean,
      median: sorted[Math.floor(n / 2)]!,
      stddev,
      min: sorted[0]!,
      max: sorted[n - 1]!,
      p5: sorted[Math.floor(n * 0.05)]!,
      p95: sorted[Math.floor(n * 0.95)]!,
      ci95: [mean - ci95Margin, mean + ci95Margin],
      perScorer,
    };
  }

  /**
   * Paired t-test: determines if the difference between two sets of scores
   * is statistically significant.
   */
  private pairedTTest(a: number[], b: number[]): { significant: boolean; pValue: number } {
    const n = Math.min(a.length, b.length);
    if (n < 2) return { significant: false, pValue: 1 };

    const diffs = a.slice(0, n).map((v, i) => b[i]! - v);
    const meanDiff = diffs.reduce((s, d) => s + d, 0) / n;
    const variance = diffs.reduce((s, d) => s + (d - meanDiff) ** 2, 0) / (n - 1);
    const se = Math.sqrt(variance / n);

    if (se === 0) return { significant: meanDiff !== 0, pValue: meanDiff === 0 ? 1 : 0 };

    const tStat = meanDiff / se;
    // Approximate p-value using normal distribution for large n
    const pValue = 2 * (1 - this.normalCDF(Math.abs(tStat)));

    return { significant: pValue < 0.05, pValue };
  }

  private normalCDF(x: number): number {
    // Abramowitz and Stegun approximation
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989422804 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }

  private timeout(ms: number): Promise<never> {
    return new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`Case timed out after ${ms}ms`)), ms),
    );
  }

  private chunk<T>(arr: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < arr.length; i += size) {
      chunks.push(arr.slice(i, i + size));
    }
    return chunks;
  }
}
