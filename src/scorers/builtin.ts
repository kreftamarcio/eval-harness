/**
 * Built-in Scorers: ready-to-use evaluation functions.
 *
 * Categories:
 *   - Exact match (deterministic)
 *   - Semantic similarity (embedding-based)
 *   - LLM-as-judge (most flexible, highest quality)
 *   - Factual accuracy (cross-reference)
 *   - Format compliance (structural validation)
 */

import type { Scorer, ScoreResult } from '../core/runner.js';

/**
 * Exact match: case-insensitive comparison with normalization.
 */
export function exactMatch(): Scorer {
  return {
    name: 'exact_match',
    description: 'Exact string match (case-insensitive, normalized whitespace)',
    async score(_input, output, expected): Promise<ScoreResult> {
      if (!expected) return { value: 0, reasoning: 'No expected output provided' };

      const normalizedOutput = normalize(output);
      const normalizedExpected = normalize(expected);

      const match = normalizedOutput === normalizedExpected;
      return {
        value: match ? 1 : 0,
        reasoning: match ? 'Exact match' : `Expected "${expected.slice(0, 100)}" but got "${output.slice(0, 100)}"`,
      };
    },
  };
}

/**
 * Contains match: checks if expected answer is contained in output.
 */
export function containsMatch(): Scorer {
  return {
    name: 'contains_match',
    description: 'Checks if expected output is contained in model response',
    async score(_input, output, expected): Promise<ScoreResult> {
      if (!expected) return { value: 0, reasoning: 'No expected output provided' };

      const contains = normalize(output).includes(normalize(expected));
      return {
        value: contains ? 1 : 0,
        reasoning: contains ? 'Output contains expected' : 'Expected not found in output',
      };
    },
  };
}

/**
 * Semantic similarity: embedding-based comparison.
 */
export function semanticSimilarity(config: {
  embed: (texts: string[]) => Promise<number[][]>;
  threshold?: number;
}): Scorer {
  const threshold = config.threshold ?? 0.8;

  return {
    name: 'semantic_similarity',
    description: `Embedding cosine similarity (threshold: ${threshold})`,
    async score(_input, output, expected): Promise<ScoreResult> {
      if (!expected) return { value: 0, reasoning: 'No expected output' };

      const embeddings = await config.embed([output, expected]);
      const similarity = cosineSimilarity(embeddings[0]!, embeddings[1]!);

      return {
        value: similarity,
        reasoning: `Cosine similarity: ${similarity.toFixed(4)} (threshold: ${threshold})`,
        metadata: { similarity, threshold, passed: similarity >= threshold },
      };
    },
  };
}

/**
 * LLM-as-judge: uses another model to evaluate quality.
 * Most flexible but slower and more expensive.
 */
export function llmJudge(config: {
  judge: (prompt: string) => Promise<string>;
  criteria: string;
  scale?: number;
}): Scorer {
  const scale = config.scale ?? 5;

  return {
    name: 'llm_judge',
    description: `LLM judge scoring (1-${scale} scale): ${config.criteria}`,
    async score(input, output, expected): Promise<ScoreResult> {
      const prompt = [
        `You are evaluating an AI response. Score it from 1 to ${scale}.`,
        ``,
        `Criteria: ${config.criteria}`,
        ``,
        `Input: ${input}`,
        expected ? `Expected: ${expected}` : '',
        `Response: ${output}`,
        ``,
        `Provide your score as a single number (1-${scale}) followed by a brief explanation.`,
        `Format: SCORE: N\nREASON: ...`,
      ].filter(Boolean).join('\n');

      const judgment = await config.judge(prompt);

      // Parse score from judge response
      const scoreMatch = judgment.match(/SCORE:\s*(\d+)/i);
      const reasonMatch = judgment.match(/REASON:\s*(.+)/is);

      const rawScore = scoreMatch ? parseInt(scoreMatch[1]!, 10) : scale / 2;
      const normalizedScore = Math.max(0, Math.min(1, (rawScore - 1) / (scale - 1)));
      const reasoning = reasonMatch?.[1]?.trim() ?? judgment.slice(0, 200);

      return {
        value: normalizedScore,
        reasoning,
        metadata: { rawScore, scale, fullJudgment: judgment },
      };
    },
  };
}

/**
 * Format compliance: checks if output matches expected format (JSON, markdown, etc.)
 */
export function formatCompliance(config: {
  format: 'json' | 'markdown' | 'code' | 'list';
  schema?: unknown;
}): Scorer {
  return {
    name: 'format_compliance',
    description: `Output format compliance check (${config.format})`,
    async score(_input, output): Promise<ScoreResult> {
      switch (config.format) {
        case 'json': {
          try {
            JSON.parse(output);
            return { value: 1, reasoning: 'Valid JSON' };
          } catch (e) {
            // Try extracting from code block
            const match = output.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
            if (match?.[1]) {
              try {
                JSON.parse(match[1]);
                return { value: 0.8, reasoning: 'Valid JSON (extracted from code block)' };
              } catch { /* fall through */ }
            }
            return { value: 0, reasoning: `Invalid JSON: ${(e as Error).message}` };
          }
        }
        case 'markdown': {
          const hasHeaders = /^#{1,6}\s/m.test(output);
          const hasFormatting = /[*_`\[\]]/m.test(output);
          const score = (hasHeaders ? 0.5 : 0) + (hasFormatting ? 0.5 : 0);
          return { value: score, reasoning: `Markdown features: headers=${hasHeaders}, formatting=${hasFormatting}` };
        }
        case 'code': {
          const hasCodeBlock = /```[\s\S]*?```/.test(output);
          return { value: hasCodeBlock ? 1 : 0, reasoning: hasCodeBlock ? 'Contains code block' : 'No code block found' };
        }
        case 'list': {
          const lines = output.split('\n').filter(l => /^[\-*\d]/.test(l.trim()));
          const score = Math.min(1, lines.length / 3);
          return { value: score, reasoning: `Found ${lines.length} list items` };
        }
      }
    },
  };
}

/**
 * Response length check: penalizes too-short or too-long responses.
 */
export function lengthCheck(config: { minChars?: number; maxChars?: number; target?: number }): Scorer {
  return {
    name: 'length_check',
    description: 'Response length appropriateness',
    async score(_input, output): Promise<ScoreResult> {
      const len = output.length;

      if (config.target) {
        // Score based on distance from target
        const ratio = Math.min(len, config.target) / Math.max(len, config.target);
        return { value: ratio, reasoning: `Length: ${len} chars (target: ${config.target})` };
      }

      if (config.minChars && len < config.minChars) {
        return { value: len / config.minChars, reasoning: `Too short: ${len} < ${config.minChars}` };
      }
      if (config.maxChars && len > config.maxChars) {
        return { value: config.maxChars / len, reasoning: `Too long: ${len} > ${config.maxChars}` };
      }

      return { value: 1, reasoning: `Length OK: ${len} chars` };
    },
  };
}

// Utilities

function normalize(text: string): string {
  return text.toLowerCase().replace(/\s+/g, ' ').trim();
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
