/**
 * Judge calibration.
 *
 * An uncalibrated LLM judge emits confident numbers of unknown validity. Measuring
 * agreement against human labels converts it from an oracle into an instrument with a
 * stated error rate, which is the difference between a score that can gate a build
 * and a score that cannot.
 *
 * Two things this reports that a single accuracy number does not:
 *
 *   1. Cohen's kappa. On a skewed label distribution, a judge that always predicts
 *      the majority class scores high RAW agreement while carrying zero information.
 *      Kappa corrects for agreement expected by chance.
 *
 *   2. Bias separated from noise. A systematically lenient judge and a randomly
 *      erratic one can share an identical mean absolute error, but the remedies are
 *      opposite: rewrite the rubric anchors, or reduce temperature and coarsen the
 *      scale. A single MAE points at neither.
 */

export interface CalibrationPair {
  id: string;
  /** Human label. Authoritative by definition. */
  humanScore: number;
  /** Judge label from a run of the rubric under test. */
  judgeScore: number;
}

export interface CalibrationConfig {
  /** Buckets used to discretise continuous scores. Defaults to 5. */
  buckets?: number;
  /** Minimum raw agreement required to gate a build. Defaults to 0.75. */
  minAgreement?: number;
  /** Minimum kappa required to gate a build. Defaults to 0.4 (moderate). */
  minKappa?: number;
}

export type JudgeStatus =
  /** May fail a build. */
  | 'gating'
  /** Reported but must not fail a build. */
  | 'advisory'
  /** Rubric does not measure what the human measures. Needs rework. */
  | 'rejected';

export interface CalibrationReport {
  n: number;
  rawAgreement: number;
  cohensKappa: number;
  kappaInterpretation: 'poor' | 'slight' | 'fair' | 'moderate' | 'substantial' | 'almost perfect';
  /** Mean signed error. Non-zero means systematic bias. */
  bias: number;
  /** SD of the signed error. Random disagreement after bias is removed. */
  noise: number;
  meanAbsoluteError: number;
  /** Pearson correlation. High r with high bias means a fixable offset. */
  correlation: number;
  /** Fraction of total error attributable to bias rather than noise. */
  biasShare: number;
  status: JudgeStatus;
  /** Actionable diagnosis, or null when the judge is well calibrated. */
  diagnosis: string | null;
  worstDisagreements: Array<{ id: string; humanScore: number; judgeScore: number; delta: number }>;
}

const DEFAULT_BUCKETS = 5;
const DEFAULT_MIN_AGREEMENT = 0.75;
const DEFAULT_MIN_KAPPA = 0.4;

export class JudgeCalibrator {
  private readonly buckets: number;
  private readonly minAgreement: number;
  private readonly minKappa: number;

  constructor(config: CalibrationConfig = {}) {
    this.buckets = config.buckets ?? DEFAULT_BUCKETS;
    this.minAgreement = config.minAgreement ?? DEFAULT_MIN_AGREEMENT;
    this.minKappa = config.minKappa ?? DEFAULT_MIN_KAPPA;

    if (this.buckets < 2) {
      throw new Error(
        `buckets must be at least 2, received ${this.buckets}. A single bucket puts ` +
          'every label in the same class, making agreement trivially 1.0 and ' +
          'meaningless.',
      );
    }
  }

  analyze(pairs: readonly CalibrationPair[]): CalibrationReport {
    if (pairs.length === 0) {
      throw new Error('Cannot calibrate against zero pairs');
    }

    const n = pairs.length;
    const signedErrors = pairs.map((p) => p.judgeScore - p.humanScore);

    const bias = this.mean(signedErrors);
    const noise = this.standardDeviation(signedErrors);
    const meanAbsoluteError = this.mean(signedErrors.map(Math.abs));

    const rawAgreement =
      pairs.filter((p) => this.bucket(p.humanScore) === this.bucket(p.judgeScore)).length / n;

    const cohensKappa = this.cohensKappa(pairs);

    const correlation = this.pearson(
      pairs.map((p) => p.humanScore),
      pairs.map((p) => p.judgeScore),
    );

    // Total squared error decomposes into bias squared plus variance. The share tells
    // you which half to attack first.
    const totalError = bias ** 2 + noise ** 2;
    const biasShare = totalError > 0 ? bias ** 2 / totalError : 0;

    const status = this.determineStatus(rawAgreement, cohensKappa);

    return {
      n,
      rawAgreement,
      cohensKappa,
      kappaInterpretation: this.interpretKappa(cohensKappa),
      bias,
      noise,
      meanAbsoluteError,
      correlation,
      biasShare,
      status,
      diagnosis: this.diagnose({ n, rawAgreement, cohensKappa, bias, noise, correlation, biasShare }),
      worstDisagreements: pairs
        .map((p) => ({
          id: p.id,
          humanScore: p.humanScore,
          judgeScore: p.judgeScore,
          delta: p.judgeScore - p.humanScore,
        }))
        .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))
        .slice(0, 10),
    };
  }

  /**
   * Cohen's kappa.
   *
   *   kappa = (p_observed - p_chance) / (1 - p_chance)
   *
   * p_chance is the agreement expected if both raters assigned labels independently
   * at their observed marginal rates. This is what makes kappa immune to the
   * majority-class trick that inflates raw agreement.
   */
  private cohensKappa(pairs: readonly CalibrationPair[]): number {
    const n = pairs.length;
    const humanCounts = new Array<number>(this.buckets).fill(0);
    const judgeCounts = new Array<number>(this.buckets).fill(0);
    let observedAgreements = 0;

    for (const pair of pairs) {
      const h = this.bucket(pair.humanScore);
      const j = this.bucket(pair.judgeScore);
      humanCounts[h]! += 1;
      judgeCounts[j]! += 1;
      if (h === j) observedAgreements += 1;
    }

    const pObserved = observedAgreements / n;

    let pChance = 0;
    for (let k = 0; k < this.buckets; k++) {
      pChance += (humanCounts[k]! / n) * (judgeCounts[k]! / n);
    }

    // Perfect chance agreement makes kappa undefined (0/0). It means one bucket holds
    // every label, so the comparison carries no information in either direction.
    if (pChance >= 1) return 0;

    return (pObserved - pChance) / (1 - pChance);
  }

  /**
   * Correct a judge score for measured systematic bias.
   *
   * Only valid when bias dominates and correlation is high: the judge is tracking the
   * human faithfully but offset. Applying this to a NOISY judge would be actively
   * harmful, shifting every score by an average that does not describe any individual
   * case, so the guard is a precondition rather than a suggestion.
   */
  applyBiasCorrection(judgeScore: number, report: CalibrationReport): number {
    if (report.biasShare < 0.5 || report.correlation < 0.7) {
      throw new Error(
        'Bias correction requires bias to dominate error (biasShare >= 0.5) and strong ' +
          `correlation (r >= 0.7). Measured biasShare=${report.biasShare.toFixed(2)}, ` +
          `r=${report.correlation.toFixed(2)}. Correcting a noisy judge shifts every ` +
          'score by an average that describes no individual case.',
      );
    }

    return Math.min(1, Math.max(0, judgeScore - report.bias));
  }

  /**
   * Continuous scores are bucketed before agreement is computed.
   *
   * Requiring exact float equality would report near-zero agreement for a judge that
   * tracks the human closely, which measures floating point rather than judgement.
   */
  private bucket(score: number): number {
    const clamped = Math.min(1, Math.max(0, score));
    return Math.min(this.buckets - 1, Math.floor(clamped * this.buckets));
  }

  private determineStatus(rawAgreement: number, kappa: number): JudgeStatus {
    // Both conditions, because either alone is gameable: high agreement with low
    // kappa is majority-class prediction, and high kappa with low agreement is a
    // judge that disagrees consistently rather than randomly.
    if (rawAgreement >= this.minAgreement && kappa >= this.minKappa) return 'gating';
    if (rawAgreement >= 0.5) return 'advisory';
    return 'rejected';
  }

  /** Landis and Koch (1977) conventional bands. Heuristics, not laws. */
  private interpretKappa(kappa: number): CalibrationReport['kappaInterpretation'] {
    if (kappa < 0.0) return 'poor';
    if (kappa < 0.2) return 'slight';
    if (kappa < 0.4) return 'fair';
    if (kappa < 0.6) return 'moderate';
    if (kappa < 0.8) return 'substantial';
    return 'almost perfect';
  }

  /**
   * Diagnosis, ordered so the most actionable finding surfaces first.
   *
   * Each branch names a specific remedy. "Agreement is low" is not a diagnosis; the
   * point is to say which of several different problems this is.
   */
  private diagnose(m: {
    n: number;
    rawAgreement: number;
    cohensKappa: number;
    bias: number;
    noise: number;
    correlation: number;
    biasShare: number;
  }): string | null {
    if (m.n < 30) {
      return (
        `n=${m.n} is too small for a stable agreement estimate. The kappa reported ` +
        'here has wide uncertainty, so treat the status as provisional.'
      );
    }

    // The specific trap raw agreement hides.
    if (m.rawAgreement >= this.minAgreement && m.cohensKappa < this.minKappa) {
      return (
        `Raw agreement is ${(m.rawAgreement * 100).toFixed(1)}% but kappa is only ` +
        `${m.cohensKappa.toFixed(2)}. The judge is likely predicting the majority ` +
        'class rather than evaluating: it agrees often because most cases share a ' +
        'label, not because it discriminates. Rebalance the calibration set, or ' +
        'coarsen the scale so the classes are genuinely distinct.'
      );
    }

    if (m.biasShare >= 0.5 && Math.abs(m.bias) > 0.05) {
      const direction = m.bias > 0 ? 'lenient' : 'harsh';
      return (
        `The judge is systematically ${direction} by ${Math.abs(m.bias).toFixed(3)}, and ` +
        `bias accounts for ${(m.biasShare * 100).toFixed(0)}% of total error with ` +
        `r=${m.correlation.toFixed(2)}. This is a rubric ANCHORING problem, not noise: ` +
        'the judge ranks cases correctly but places the whole scale off centre. ' +
        'Rewrite the scale descriptions with concrete examples at each level.'
      );
    }

    if (m.noise > 0.2) {
      return (
        `Random disagreement is high (SD ${m.noise.toFixed(3)}) while bias is only ` +
        `${m.bias.toFixed(3)}. The judge is inconsistent rather than offset. Lower the ` +
        'temperature, reduce the number of scale points, or split a compound rubric ' +
        'into separate single-criterion judgements.'
      );
    }

    if (m.correlation < 0.5) {
      return (
        `Correlation with human labels is only r=${m.correlation.toFixed(2)}. The judge ` +
        'is not measuring the same construct as the human. This needs a rubric ' +
        'rewrite, not a parameter tweak.'
      );
    }

    return null;
  }

  private mean(values: readonly number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  /** Sample SD with Bessel's correction. */
  private standardDeviation(values: readonly number[]): number {
    const n = values.length;
    if (n < 2) return 0;

    const m = this.mean(values);
    const sumSquares = values.reduce((sum, v) => sum + (v - m) ** 2, 0);
    return Math.sqrt(sumSquares / (n - 1));
  }

  private pearson(a: readonly number[], b: readonly number[]): number {
    const n = a.length;
    if (n < 2) return 0;

    const meanA = this.mean(a);
    const meanB = this.mean(b);

    let covariance = 0;
    let varianceA = 0;
    let varianceB = 0;

    for (let i = 0; i < n; i++) {
      const devA = a[i]! - meanA;
      const devB = b[i]! - meanB;
      covariance += devA * devB;
      varianceA += devA ** 2;
      varianceB += devB ** 2;
    }

    // Zero variance on either side makes r undefined. It means one rater gave every
    // case an identical score, which is a degenerate rubric rather than a
    // correlation of zero.
    const denominator = Math.sqrt(varianceA * varianceB);
    return denominator === 0 ? 0 : covariance / denominator;
  }
}
