# Ancileo x MSIG Conversational Insurance Platform

Aurora is a multi-channel conversational concierge that turns travel insurance into an intelligent dialogue. The system combines Groq-hosted language models with domain-aware tools for policy reasoning, claims-driven recommendations, document intelligence, and seamless payments that plug into any customer touchpoint.

## Key Capabilities
- **Agentic orchestration** - Groq LLM governs the flow, decides when to call tools, keeps empathetic tone, and cites sources.
- **Policy intelligence** - Automatic indexing of the provided policy PDFs into a persistent vector store for grounded answers.
- **Predictive insights** - Claims analytics surface risk patterns and plan recommendations tailored to itinerary context.
- **Document intelligence** - Booking PDFs can be parsed to extract travellers, destinations, schedules, and spend signals.
- **Seamless commerce** - Stripe payments or the supplied Docker stack create and monitor checkout sessions inside the chat.
- **Channel adapters** - Web, WhatsApp, and Telegram endpoints reuse the same orchestration core, keeping journeys consistent.
- **Partner portal ready** - A demo website lets distribution partners log in, review traveller history, and launch Aurora with pre-filled context.

## Project Layout
```
conversational_insurance_agent/
|- .env                    # Environment configuration (fill in secrets!)
|- .gitignore
|- README.md
|- requirements.txt
|- src/
   |- main.py             # FastAPI application entrypoint
   |- config.py           # Settings loader
   |- core/               # Orchestrator and tool registry
   |- services/           # External integrations (payments)
   |- tools/              # Domain tools for policy RAG, claims, docs
   |- channels/           # Channel message helpers
   |- state/              # Redis-backed conversation memory
```

## Getting Started
1. **Install dependencies**
   ```bash
   cd conversational_insurance_agent
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment** - update `.env` with valid credentials.
   - `GROQ_API_KEY` and (optionally) `GROQ_MODEL`
   - `GROQ_VISION_MODEL` (defaults to `llama-3.2-90b-vision-preview`)
   - `STRIPE_API_KEY` and `STRIPE_WEBHOOK_SECRET` (leave blank to rely on the provided payments microservice)
   - `REDIS_URL`, `VECTOR_DB_PATH`, `CLAIMS_DATA_PATH` as needed
   - `CLAIMS_DB_HOST`, `CLAIMS_DB_PORT`, `CLAIMS_DB_NAME`, `CLAIMS_DB_USER`, `CLAIMS_DB_PASSWORD`, `CLAIMS_DB_TABLE` if you prefer training the risk predictor directly from Postgres
   - `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_WHATSAPP_NUMBER` for WhatsApp (or `TWILIO_API_KEY`/`TWILIO_API_SECRET` if you prefer API keys)
   - `ANCILEO_API_KEY` to access the Ancileo pricing & purchase APIs (override `ANCILEO_BASE_URL` and `ANCILEO_DEFAULT_*` for non-default environments)
   - `SESSION_SECRET` to secure the partner portal cookies (defaults to a development fallback)

3. **Prepare Redis**
   ```bash
   docker run -d -p 6379:6379 redis:7
   ```

4. **Seed the policy index** - first launch will index PDFs from `Policy_Wordings/`; you can force a rebuild later:
   ```bash
   curl -X POST http://localhost:8080/tools/policy/index
   ```

5. **Claims analytics input** - convert `Claims_Data_DB.pdf` into CSV or Parquet and place it at `data/claims_stats.parquet`, or update `CLAIMS_DATA_PATH`.

6. **Run the API**
   ```bash
   uvicorn src.main:app --reload --port 8080
   ```

## Core Endpoints
- `POST /chat` - channel-agnostic message handler (web, mobile, partner apps)
- `POST /webhooks/whatsapp` - Twilio webhook adapter (consumes `application/x-www-form-urlencoded` payloads with optional media)
- `POST /webhooks/telegram` - Telegram bot webhook adapter
- `POST /tools/policy/index` - retained for backwards compatibility; policy agent loads taxonomy data on demand and returns a `skipped` status
- `GET /healthz` - service readiness check
- `GET /integration/login` - partner portal sign-in screen (demo credentials provided)
- `POST /integration/chat/send` - lightweight bridge that injects traveller data from the partner portal into the orchestrator before chatting

## Partner Integration Portal
1. Start the FastAPI server as usual (`uvicorn src.main:app --reload --port 8080`).
2. Visit `http://localhost:8080/integration/login` and sign in with one of the demo accounts shown on the login page (e.g. `alice@example.com / travel123`).
3. Review the traveller profile, upcoming trip, and historical policy purchases surfaced from the mock database.
4. Chat with Aurora inside the web console?each message automatically merges the known traveller data into the orchestrator so the assistant can confirm details, run claims intelligence, and recommend the right plan immediately.

The portal exercises the "seamless integration" workflow: when a traveller profile is complete, Aurora confirms it and jumps straight into tool calls; when data is sparse, the orchestrator instructs the model to gather missing fields before recommending coverage and to re-verify information prior to payment.

## Tool Registry
| Tool               | Purpose                                                              |
|--------------------|----------------------------------------------------------------------|
| `policy_research`  | LangGraph agent that surfaces eligible benefits from the taxonomy    |
| `claims_recommendation` | Generate risk-aware plan suggestions using historical claims   |
| `document_ingest`  | Parse itineraries and bookings for traveller, date, and cost signals |
| `payment_checkout` | Create a checkout session via Stripe or the hackathon payments stack |
| `payment_status`   | Poll the latest status of a checkout session                         |
| `travel_insurance_purchase` | Finalise policy issuance with Ancileo after payment success |

<!-- `travel_insurance_quote` tool temporarily disabled; LLM now provides pricing guidance -->

The orchestrator instructs the LLM to emit JSON whenever a tool call is required, executes the tool, then resumes the conversation with the result to maintain fluid dialogue.

## Channel Integrations
- **Web / Mobile** - call `POST /chat` with a stable `session_id` per user/device.
- **WhatsApp** - expose `/webhooks/whatsapp`, configure Twilio to forward inbound messages, and provide Twilio credentials so media can be downloaded. Image/PDF attachments are summarised with Groq before being forwarded to the orchestrator.
- **Telegram** - point Bot API webhook to `/webhooks/telegram`; `channels/telegram.py` helps translate raw updates.

Persisting the session ID ensures Redis-backed memory keeps context even when users switch devices or channels.

## Payments Flow
1. Agent derives a pricing recommendation within the conversation (Ancileo pricing API temporarily unavailable) and explains the assumptions.
2. Agent calls `payment_checkout` with the agreed plan_code, price in minor units, and success/cancel URLs. Metadata should include any available `quoteId`, `offerId`, and traveller references.
3. Service attempts to use the provided payments microservice (`Payments/`). If unavailable, it falls back to direct Stripe Checkout using configured keys.
4. Poll `payment_status` or let the webhook update the conversation before delivering policy documents; once marked successful, call `travel_insurance_purchase` to finalise issuance with Ancileo.

## Extending the Platform
- Register new `ToolSpec` entries in `core/setup.py` for loyalty, ancillaries, or external data sources.
- Extend the policy research agent with richer LangGraph workflows or additional data connectors.
- Enhance memory by extending the Redis session store or introducing Temporal/LangGraph workflows.
- Wrap Groq calls with Langfuse/Helicone for observability and guardrail enforcement.

## Testing Ideas
- Unit test the tools (policy research agent, claims analytics, document ingestion) with fixtures from the repository.
- Add integration tests around `/chat` using mocked Groq responses to validate tool-calling logic.
- Exercise payments end-to-end via `Payments/test_payment_flow.py` and Stripe test cards.

## Roadmap Suggestions
- Voice and image ingestion for itineraries.
- Emotion-aware tone modulation and escalation triggers.
- Real-time push updates (WebSockets or server-sent events) for payment completion.
- Continuous learning loop: capture unresolved questions, retrain or augment retrieval prompts automatically.

---
Fill in your Groq and Stripe credentials inside `.env`, start the FastAPI service, and Aurora is ready to offer personalised, citation-backed travel insurance guidance across every channel.
