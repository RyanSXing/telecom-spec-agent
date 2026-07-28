"use client";

import { FormEvent, useState } from "react";

type Citation = {
  chunk_id: string;
  spec_id: string;
  release: string;
  clause_number: string;
  title: string;
  quoted_span: string;
};

type Answer = {
  answer: string;
  citations: Citation[];
  confidence: "high" | "medium" | "low";
  unanswered_aspects: string[];
};

const examples = [
  "What are service-based interfaces used for?",
  "What is a QoS Flow?",
  "Summarize clause 4.2.6.",
];

const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function Home() {
  const [question, setQuestion] = useState(examples[0]);
  const [result, setResult] = useState<Answer | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) return;

    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${apiUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });
      if (!response.ok) {
        throw new Error(
          response.status === 503
            ? "Search or Vertex AI is unavailable. Check the local services."
            : "The query could not be completed.",
        );
      }
      setResult((await response.json()) as Answer);
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : "The query could not be completed.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header className="site-header">
        <a className="wordmark" href="#top" aria-label="Spec Copilot home">
          <span aria-hidden="true">SC</span>
          Spec Copilot
        </a>
        <div className="source-badge">
          <span aria-hidden="true" />
          3GPP TS 23.501 · Rel-17
        </div>
      </header>

      <section className="hero" id="top">
        <p className="eyebrow">Telecom architecture intelligence</p>
        <h1>
          Ask the specification.
          <br />
          <span>Verify every claim.</span>
        </h1>
        <p className="intro">
          Hybrid retrieval over clause-structured 5G documentation, with answers grounded in
          exact source passages.
        </p>
      </section>

      <section className="workspace" aria-label="Specification query workspace">
        <form onSubmit={submit}>
          <label htmlFor="question">Question for TS 23.501</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            maxLength={2000}
            rows={4}
            placeholder="Ask about 5G architecture, network functions, or an exact clause…"
          />
          <div className="form-footer">
            <span>{question.length} / 2,000</span>
            <button disabled={loading || !question.trim()} type="submit">
              {loading ? "Retrieving…" : "Search specification"}
            </button>
          </div>
        </form>

        <div className="examples" aria-label="Example questions">
          <p>Try an example</p>
          <div>
            {examples.map((example) => (
              <button key={example} type="button" onClick={() => setQuestion(example)}>
                {example}
              </button>
            ))}
          </div>
        </div>
      </section>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {result && (
        <section className="answer" aria-live="polite">
          <div className="answer-heading">
            <p>Grounded answer</p>
            <span className={`confidence confidence-${result.confidence}`}>
              {result.confidence} confidence
            </span>
          </div>
          <p className="answer-copy">{result.answer}</p>

          {result.unanswered_aspects.length > 0 && (
            <aside>
              <strong>Not fully answered</strong>
              <ul>
                {result.unanswered_aspects.map((aspect) => (
                  <li key={aspect}>{aspect}</li>
                ))}
              </ul>
            </aside>
          )}

          <div className="citations">
            <h2>Source evidence</h2>
            {result.citations.map((citation) => (
              <details key={`${citation.chunk_id}-${citation.quoted_span}`}>
                <summary>
                  <span>
                    {citation.spec_id} §{citation.clause_number}
                  </span>
                  {citation.title}
                </summary>
                <blockquote>{citation.quoted_span}</blockquote>
                <small>
                  {citation.release} · chunk {citation.chunk_id.slice(0, 10)}
                </small>
              </details>
            ))}
          </div>
        </section>
      )}

      <footer>
        Public 3GPP documentation · Clause-aware indexing · OpenSearch + Vertex AI
      </footer>
    </main>
  );
}
