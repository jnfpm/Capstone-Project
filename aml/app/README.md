# app/ — the assistant web app (next step)

The deliverable the course asks for: a deployed, conversational AI assistant with function-calling tools,
document understanding and a domain-specific tool suite. It sits on top of `../agents/`:

- the user (an analyst) opens an alert or asks about an account / transfer;
- the app calls the agents, which call the scorer (`src/sample_modeling.score_transactions`) and the
  lookup tools on `data/processed/hi_small_joined.parquet`;
- it shows the score, the model's reasons and the agent's case write-up.

The app must never read the ground-truth columns (`Is Laundering`, `Attempt ID`, `Pattern Type`).
Configuration (API keys) goes in a `.env` file, which is not committed.
