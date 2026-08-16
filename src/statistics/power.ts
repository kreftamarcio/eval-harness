/**
 * Statistical power analysis.
 *
 * Runs BEFORE any paid inference and answers one question: can this dataset detect
 * the difference we care about at all?
 *
 * The alternative is spending money on a run that mathematically cannot answer the
 * question, then answering it anyway from noise. That is the single most common way
 * eval pipelines produce confident wrong conclusions, and it is entirely avoidable
 * with arithmetic that costs nothing.
 *
 * Paired-design sample size:
 *
 *   n = ( (z_{alpha/2} + z_beta)^2 * sigma^2_diff ) / d^2
 *
 * Solved for d, this gives the minimum detectable effect at fixed n.
 */

export interface PowerConfig {
  /** Significance level. Defaults to 0.05. */
  alpha?: number;
  /** Desired power, the probability of detecting a true effect. Defaults to 0.8. */
  power?: number;
}

export interface PowerAnalysis {
  n: number;
  /** SD of per-example score differences. */
  stdDevOfDifferences: number;
  alpha: number;
  power: number;
  /** Smallest effect this dataset can resolve, in score units. */
  minimumDetectableEffect: number;
  /** Verdict against a target, when one was supplied. */
  target?: {
    effect: number;
    sufficient: boolean;
    requiredN: number;
    additionalExamplesNeeded: number;
    /** Achieved power at the current n for the target effect. */
    achievedPower: number;
  };
  /** Present when the dataset is too small to support any conclusion. */
  warning: string | null;
}

export interface SlicePowerAnalysis {
  tag: string;
  n: number;
  minimumDetectableEffect: number;
  /** True when this slice cannot detect the effect the overall dataset can. */
  underpowered: boolean;
}

export interface CostProjection {
  currentN: number;
  requiredN: number;
  additionalExamples: number;
  costPerExampleUSD: number;
  additionalCostUSD: number;
  /** Cost of one full comparison run at the required size, both configurations. */
  perRunCostUSD: number;
}

const DEFAULT_ALPHA = 0.05;
const DEFAULT_POWER = 0.8;

/** Below this, an eval is a smoke test rather than evidence. */
const MINIMUM_USEFUL_N = 30;

export class PowerAnalyzer {
  private readonly alpha: number;
  private readonly targetPower: number;

  constructor(config: PowerConfig = {}) {
    this.alpha = config.alpha ?? DEFAULT_ALPHA;
    this.targetPower = config.power ?? DEFAULT_POWER;

    if (this.alpha <= 0 || this.alpha >= 1) {
      throw new RangeError(`alpha must be in (0,1), received ${this.alpha}`);
    }
    if (this.targetPower <= 0 || this.targetPower >= 1) {
      throw new RangeError(`power must be in (0,1), received ${this.targetPower}`);
    }
  }

  analyze(params: { n: number; stdDevOfDifferences: number; targetEffect?: number }): PowerAnalysis {
    const { n, stdDevOfDifferences: sd } = params;

    if (n < 2) {
      throw new Error(`Power analysis requires at least 2 examples, received ${n}`);
    }
    if (sd <= 0) {
      throw new Error(
        `stdDevOfDifferences must be positive, received ${sd}. A zero SD would imply ` +
          'every example differs by exactly the same amount, which never happens with ' +
          'a stochastic model and usually means the estimate is wrong.',
      );
    }

    const mde = this.minimumDetectableEffect(n, sd);

    const analysis: PowerAnalysis = {
      n,
      stdDevOfDifferences: sd,
      alpha: this.alpha,
      power: this.targetPower,
      minimumDetectableEffect: mde,
      warning:
        n < MINIMUM_USEFUL_N
          ? `n=${n} has essentially no statistical power. This is a smoke test, not an ` +
            'eval. Treating its output as evidence invites conclusions it cannot support.'
          : null,
    };

    if (params.targetEffect !== undefined) {
      const requiredN = this.requiredSampleSize(params.targetEffect, sd);

      analysis.target = {
        effect: params.targetEffect,
        sufficient: mde <= params.targetEffect,
        requiredN,
        additionalExamplesNeeded: Math.max(0, requiredN - n),
        achievedPower: this.achievedPower(n, params.targetEffect, sd),
      };
    }

    return analysis;
  }

  /**
   * Per-slice power.
   *
   * This is the trap that catches teams who DID run a power analysis. A 240-example
   * dataset has reasonable power overall and almost none on a 12-example slice, so a
   * business-critical slice can collapse without the regression gate detecting it.
   * Overall power is not slice power.
   */
  analyzeSlices(
    slices: ReadonlyArray<{ tag: string; n: number }>,
    stdDevOfDifferences: number,
    overallMde: number,
  ): SlicePowerAnalysis[] {
    return slices
      .map((slice) => {
        const mde =
          slice.n >= 2
            ? this.minimumDetectableEffect(slice.n, stdDevOfDifferences)
            : Number.POSITIVE_INFINITY;

        return {
          tag: slice.tag,
          n: slice.n,
          minimumDetectableEffect: mde,
          underpowered: mde > overallMde * 1.5,
        };
      })
      .sort((a, b) => b.minimumDetectableEffect - a.minimumDetectableEffect);
  }

  /**
   * Cost of closing a power gap.
   *
   * "Add 400 examples" is only actionable with the dollar figure attached. The
   * per-run figure doubles the per-example cost because a paired comparison runs
   * every example through both configurations.
   */
  projectCost(params: {
    currentN: number;
    targetEffect: number;
    stdDevOfDifferences: number;
    costPerExampleUSD: number;
  }): CostProjection {
    const requiredN = this.requiredSampleSize(params.targetEffect, params.stdDevOfDifferences);
    const additional = Math.max(0, requiredN - params.currentN);

    return {
      currentN: params.currentN,
      requiredN,
      additionalExamples: additional,
      costPerExampleUSD: params.costPerExampleUSD,
      additionalCostUSD: additional * params.costPerExampleUSD,
      perRunCostUSD: requiredN * params.costPerExampleUSD * 2,
    };
  }

  /**
   * Estimate the SD of differences from a pilot run.
   *
   * Worth doing rather than guessing: the SD dominates every number here, and a
   * conservative guess produces a conservative sample size while an optimistic one
   * produces a run that cannot answer the question.
   */
  estimateStdDev(baselineScores: readonly number[], candidateScores: readonly number[]): number {
    if (baselineScores.length !== candidateScores.length) {
      throw new Error(
        `Paired estimation requires equal-length arrays. Got ${baselineScores.length} ` +
          `and ${candidateScores.length}. Align by example id before estimating.`,
      );
    }
    if (baselineScores.length < 2) {
      throw new Error('Need at least 2 paired observations to estimate a standard deviation');
    }

    const differences = candidateScores.map((c, i) => c - baselineScores[i]!);
    const mean = differences.reduce((a, b) => a + b, 0) / differences.length;
    const sumSquares = differences.reduce((sum, d) => sum + (d - mean) ** 2, 0);

    // Bessel's correction: this is a sample SD used to project a population.
    return Math.sqrt(sumSquares / (differences.length - 1));
  }

  minimumDetectableEffect(n: number, sd: number): number {
    const zAlpha = this.normalQuantile(1 - this.alpha / 2);
    const zBeta = this.normalQuantile(this.targetPower);
    return ((zAlpha + zBeta) * sd) / Math.sqrt(n);
  }

  requiredSampleSize(effect: number, sd: number): number {
    if (effect <= 0) {
      throw new Error(`Target effect must be positive, received ${effect}`);
    }

    const zAlpha = this.normalQuantile(1 - this.alpha / 2);
    const zBeta = this.normalQuantile(this.targetPower);
    return Math.ceil(((zAlpha + zBeta) ** 2 * sd ** 2) / effect ** 2);
  }

  /**
   * Power actually achieved at a given n for a given effect.
   *
   *   power = Phi( d*sqrt(n)/sigma - z_{alpha/2} )
   *
   * Reported because "insufficient" is not equally bad everywhere: 0.78 power is
   * usually worth running, 0.31 is not, and a boolean cannot express the difference.
   */
  achievedPower(n: number, effect: number, sd: number): number {
    const zAlpha = this.normalQuantile(1 - this.alpha / 2);
    const z = (effect * Math.sqrt(n)) / sd - zAlpha;
    return this.normalCdf(z);
  }

  /**
   * Inverse normal CDF via Acklam's rational approximation.
   *
   * A lookup table for the common alphas would be shorter, but silently
   * mis-answering a non-standard alpha is worse than the arithmetic. Absolute error
   * is below 1.15e-9, far tighter than the uncertainty in the SD estimate feeding it.
   */
  private normalQuantile(p: number): number {
    if (p <= 0 || p >= 1) {
      throw new RangeError(`normalQuantile requires 0 < p < 1, received ${p}`);
    }

    const a = [
      -3.969683028665376e1, 2.20946098424521e2, -2.759285104469687e2, 1.38357751867269e2,
      -3.066479806614716e1, 2.506628277459239,
    ];
    const b = [
      -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1,
      -1.328068155288572e1,
    ];
    const c = [
      -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734,
      4.374664141464968, 2.938163982698783,
    ];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];

    const pLow = 0.02425;

    if (p < pLow) {
      const q = Math.sqrt(-2 * Math.log(p));
      return (
        (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
        ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
      );
    }

    if (p > 1 - pLow) {
      const q = Math.sqrt(-2 * Math.log(1 - p));
      return (
        -(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
        ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
      );
    }

    const q = p - 0.5;
    const r = q * q;
    return (
      ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) /
      (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1)
    );
  }

  /**
   * Normal CDF via Abramowitz and Stegun 7.1.26 applied to erf.
   * Absolute error below 1.5e-7, which is ample for reporting power to two decimals.
   */
  private normalCdf(z: number): number {
    return 0.5 * (1 + this.erf(z / Math.SQRT2));
  }

  private erf(x: number): number {
    const sign = x < 0 ? -1 : 1;
    const absX = Math.abs(x);

    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    const t = 1 / (1 + p * absX);
    const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-absX * absX);

    return sign * y;
  }
}
