/**
 * Bootstrap resampling for confidence intervals and significance testing.
 *
 * Why bootstrap instead of parametric tests:
 * LLM evaluation scores routinely violate normality assumptions. They are
 * bounded (often [0,1]), frequently bimodal (a case is either handled or it
 * isn't), and skewed. The bootstrap makes no distributional assumption: it
 * estimates the sampling distribution by resampling the observed data.
 *
 * Reference: Efron & Tibshirani, "An Introduction to the Bootstrap" (1993).
 */

export interface ConfidenceInterval {
  lower: number;
  upper: number;
  level: number;
  pointEstimate: number;
}

export interface PairedComparison {
  meanDifference: number;
  confidenceInterval: ConfidenceInterval;
  /** Bootstrap p-value: proportion of resamples where the sign flips. */
  pValue: number;
  significant: boolean;
  /** Cohen's d for paired samples. */
  effectSize: number;
  effectSizeLabel: 'negligible' | 'small' | 'medium' | 'large';
  sampleSize: number;
}

export interface BootstrapConfig {
  iterations: number;
  confidenceLevel: number;
  /** Optional seed for reproducible resampling. */
  seed?: number;
}

const DEFAULT_CONFIG: BootstrapConfig = {
  iterations: 10_000,
  confidenceLevel: 0.95,
};

export class Bootstrap {
  private readonly config: BootstrapConfig;
  private rngState: number;

  constructor(config: Partial<BootstrapConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    if (this.config.iterations < 1000) {
      throw new Error(
        'Bootstrap iterations below 1000 produce unstable interval estimates. ' +
        'Use at least 1000, preferably 10000.',
      );
    }
    if (this.config.confidenceLevel <= 0 || this.config.confidenceLevel >= 1) {
      throw new Error('confidenceLevel must be strictly between 0 and 1');
    }

    this.rngState = config.seed ?? Date.now();
  }

  /**
   * Percentile bootstrap confidence interval for the mean.
   *
   * Algorithm:
   *   1. Resample n values with replacement from the original n values.
   *   2. Compute the mean of the resample.
   *   3. Repeat B times to build the bootstrap distribution of the mean.
   *   4. Take the alpha/2 and 1-alpha/2 percentiles of that distribution.
   */
  meanConfidenceInterval(scores: number[]): ConfidenceInterval {
    if (scores.length === 0) {
      throw new Error('Cannot compute a confidence interval over zero samples');
    }
    if (scores.length < 5) {
      // Not an error, but the caller should know the interval will be very wide.
      // We still compute it rather than silently refusing.
    }

    const bootstrapMeans = new Float64Array(this.config.iterations);

    for (let i = 0; i < this.config.iterations; i++) {
      bootstrapMeans[i] = this.resampleMean(scores);
    }

    const sorted = Float64Array.prototype.slice.call(bootstrapMeans).sort((a, b) => a - b);
    const alpha = 1 - this.config.confidenceLevel;

    return {
      lower: this.percentile(sorted, alpha / 2),
      upper: this.percentile(sorted, 1 - alpha / 2),
      level: this.config.confidenceLevel,
      pointEstimate: this.mean(scores),
    };
  }

  /**
   * Paired comparison between two configurations evaluated on the same dataset.
   *
   * Pairing matters: per-example difficulty is a large variance component.
   * By comparing differences within each example, that variance cancels out,
   * making the test substantially more sensitive than an unpaired comparison
   * at the same sample size.
   */
  comparePaired(baseline: number[], candidate: number[]): PairedComparison {
    if (baseline.length !== candidate.length) {
      throw new Error(
        `Paired comparison requires equal-length arrays. ` +
        `Got baseline=${baseline.length}, candidate=${candidate.length}. ` +
        `If the runs covered different examples, align them by example id first.`,
      );
    }
    if (baseline.length === 0) {
      throw new Error('Cannot compare zero samples');
    }

    // Per-example differences
    const differences = candidate.map((c, i) => c - baseline[i]!);
    const observedMeanDiff = this.mean(differences);

    // Bootstrap the distribution of the mean difference
    const bootstrapDiffs = new Float64Array(this.config.iterations);
    for (let i = 0; i < this.config.iterations; i++) {
      bootstrapDiffs[i] = this.resampleMean(differences);
    }

    const sorted = Float64Array.prototype.slice.call(bootstrapDiffs).sort((a, b) => a - b);
    const alpha = 1 - this.config.confidenceLevel;

    const ci: ConfidenceInterval = {
      lower: this.percentile(sorted, alpha / 2),
      upper: this.percentile(sorted, 1 - alpha / 2),
      level: this.config.confidenceLevel,
      pointEstimate: observedMeanDiff,
    };

    // Two-sided bootstrap p-value: how often does the resampled difference
    // fall on the opposite side of zero from the observed difference?
    let crossings = 0;
    for (let i = 0; i < sorted.length; i++) {
      const value = sorted[i]!;
      if (observedMeanDiff >= 0 ? value <= 0 : value >= 0) {
        crossings++;
      }
    }
    const pValue = Math.min(1, (2 * crossings) / this.config.iterations);

    const effectSize = this.cohensDPaired(differences);

    return {
      meanDifference: observedMeanDiff,
      confidenceInterval: ci,
      pValue,
      // Significant when the interval excludes zero.
      significant: ci.lower > 0 || ci.upper < 0,
      effectSize,
      effectSizeLabel: this.labelEffectSize(effectSize),
      sampleSize: baseline.length,
    };
  }

  /**
   * Cohen's d for paired samples: mean difference over the standard deviation
   * of the differences.
   */
  private cohensDPaired(differences: number[]): number {
    const meanDiff = this.mean(differences);
    const sd = this.standardDeviation(differences);
    return sd === 0 ? 0 : meanDiff / sd;
  }

  /**
   * Conventional thresholds (Cohen, 1988). These are heuristics, not laws:
   * a "small" effect on a metric that matters can still be worth shipping.
   */
  private labelEffectSize(d: number): PairedComparison['effectSizeLabel'] {
    const abs = Math.abs(d);
    if (abs < 0.2) return 'negligible';
    if (abs < 0.5) return 'small';
    if (abs < 0.8) return 'medium';
    return 'large';
  }

  private resampleMean(values: number[]): number {
    const n = values.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const index = Math.floor(this.nextRandom() * n);
      sum += values[index]!;
    }
    return sum / n;
  }

  private percentile(sortedValues: number[] | Float64Array, p: number): number {
    const n = sortedValues.length;
    if (n === 0) return 0;

    const index = p * (n - 1);
    const lowerIndex = Math.floor(index);
    const upperIndex = Math.ceil(index);

    if (lowerIndex === upperIndex) {
      return sortedValues[lowerIndex]!;
    }

    // Linear interpolation between adjacent order statistics
    const weight = index - lowerIndex;
    return sortedValues[lowerIndex]! * (1 - weight) + sortedValues[upperIndex]! * weight;
  }

  private mean(values: number[]): number {
    if (values.length === 0) return 0;
    let sum = 0;
    for (const v of values) sum += v;
    return sum / values.length;
  }

  private standardDeviation(values: number[]): number {
    const n = values.length;
    if (n < 2) return 0;

    const m = this.mean(values);
    let sumSquaredDeviations = 0;
    for (const v of values) {
      const deviation = v - m;
      sumSquaredDeviations += deviation * deviation;
    }

    // Bessel's correction: n-1 for the sample standard deviation
    return Math.sqrt(sumSquaredDeviations / (n - 1));
  }

  /**
   * mulberry32: small, fast, deterministic PRNG.
   * Deterministic seeding matters here because an evaluation result that
   * changes between identical runs is not reproducible, and reproducibility
   * is the entire point of a harness.
   */
  private nextRandom(): number {
    this.rngState = (this.rngState + 0x6D2B79F5) | 0;
    let t = this.rngState;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}
