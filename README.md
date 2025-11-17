# H-LLM Multi-Model REST API

FastAPI backend for H-LLM Multi-Model iOS app.

## Features

- **Authentication**: Twilio SMS OTP verification with JWT tokens
- **Multi-Model AI**: Query 8 AI models simultaneously
  - OpenAI GPT-4o
  - Anthropic Claude Sonnet 4.5
  - Google Gemini 2.5 Flash
  - Cohere Command R
  - DeepSeek
  - OpenRouter
  - Perplexity
  - xAI Grok
- **H-Score**: Hallucination risk calculation (0-100)
- **Red/Blue/Purple Team Analysis**: Security-style AI response evaluation

## API Endpoints

### Authentication

#### Send OTP
```
POST /api/auth/send-otp
{
  "phone_number": "+12345678900"
}
```

#### Verify OTP
```
POST /api/auth/verify-otp
{
  "phone_number": "+12345678900",
  "code": "123456"
}
```

Returns:
```json
{
  "success": true,
  "tokens": {
    "access": "eyJ...",
    "refresh": "eyJ..."
  },
  "user": {
    "phone": "+12345678900",
    "id": "user_...",
    "subscription_tier": "free"
  }
}
```

### Query

#### Multi-Model Query
```
POST /api/query
Headers: Authorization: Bearer <access_token>
{
  "query": "What is the capital of France?",
  "enable_rag": true,
  "enable_red_team": true,
  "enable_blue_team": true,
  "enable_purple_team": true,
  "show_metadata": true
}
```

Returns:
```json
{
  "success": true,
  "responses": [
    {
      "model": "OpenAI",
      "response": "The capital of France is Paris.",
      "metadata": {}
    }
  ],
  "h_score": {
    "final": 8.5,
    "safety": 9.0,
    "trust": 8.5,
    "confidence": 8.0,
    "quality": 10.0
  },
  "team_analysis": {
    "red_team": "Risk Score: 2/10...",
    "blue_team": "Trust Score: 8/10...",
    "purple_team": "Confidence Score: 8/10..."
  }
}
```

### Health Check

```
GET /health
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run server:
```bash
python main.py
# Or with uvicorn:
uvicorn main:app --reload
```

4. Access API docs:
```
http://localhost:8000/docs
```

## Development

- Development mode: Use code `123456` for OTP verification when Twilio is not configured
- Swagger docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deployment

Deploy to Render.com:

1. Connect GitHub repository
2. Set environment variables in Render dashboard
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## TODO

- [ ] Connect to Supabase database
- [ ] Implement user subscription management
- [ ] Add Apple In-App Purchase receipt validation
- [ ] Add rate limiting
- [ ] Add request caching
- [ ] Add RAG (Retrieval-Augmented Generation)
