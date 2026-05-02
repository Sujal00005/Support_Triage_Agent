# Multi-Domain Support Triage Agent

> 🏆 Built for the **HackerRank Agentic AI Hackathon**

A terminal-based support triage agent that handles support tickets across three ecosystems:
- **HackerRank** — https://support.hackerrank.com
- **Claude** — https://support.claude.com/en/
- **Visa** — https://www.visa.co.in/support.html

---

## How It Works

For each support ticket the agent:
1. **Identifies the domain** — uses the `company` field or infers it via keyword scoring when `company` is `None`
2. **Detects cross-domain confusion** — if the ticket content belongs to a different domain than declared, it auto-corrects
3. **Detects multiple requests** — flags tickets containing more than one distinct issue
4. **Runs safety pre-classification** — catches prompt injection, malicious requests, sensitive triggers, and out-of-scope topics before the LLM
5. **Retrieves relevant corpus** — uses TF-IDF weighted scoring to extract the most relevant paragraphs from the scraped support corpus
6. **Classifies and responds** — sends the ticket + corpus snippet to an LLM with strict grounding rules
7. **Validates output** — sanitizes all fields before writing to CSV

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your API key
```bash
cp .env.example .env
# Edit .env and add your Groq API key (free at console.groq.com)
```

### 3. Scrape the support corpus (one-time)
```bash
python scrape_corpus.py
```
Creates `corpus/hackerrank.txt`, `corpus/claude.txt`, `corpus/visa.txt` from the live support sites.

### 4. Place your input CSV
Put `support_tickets.csv` in the project root.

### 5. Run the agent
```bash
python agent.py
```
Outputs `output.csv` and `log.txt`.

---

## File Structure
```
.
├── agent.py                  # Main triage agent
├── scrape_corpus.py          # One-time corpus scraper
├── scrape_hackerrank_only.py # Re-scrape HackerRank + Claude only
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── corpus/
│   ├── hackerrank.txt        # Scraped HackerRank support docs
│   ├── claude.txt            # Scraped Claude support docs
│   └── visa.txt              # Scraped Visa support docs
├── support_tickets.csv       # Input tickets
├── output.csv                # Agent predictions
└── log.txt                   # Full run transcript
```

---

## Output Schema
| Field | Allowed Values |
|-------|---------------|
| `status` | `replied`, `escalated` |
| `product_area` | Support category string |
| `response` | User-facing answer grounded in corpus |
| `justification` | Internal reasoning + corpus source + retrieval confidence |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |

---

## Key Features

### TF-IDF Weighted Retrieval
Rare domain-specific terms (e.g. "chargeback", "proctoring", "impersonation") score higher than common words, ensuring the most relevant corpus paragraphs are selected for each ticket.

### Multi-Layer Safety Pre-Classifier
Runs before the LLM and catches:
- **Prompt injection** — instruction overrides, identity manipulation, information extraction attempts, jailbreaks (categorized by type)
- **Malicious requests** — harmful code requests, system attacks
- **Sensitive triggers** — identity theft, fraud, security vulnerabilities, subscription changes
- **Out-of-scope topics** — irrelevant or off-domain queries
- **Vague tickets** — requests too ambiguous to classify

### Cross-Domain Confusion Detection
Detects when a ticket's content belongs to a different domain than the declared `company` field and automatically corrects the routing.

### Multi-Request Detection
Flags tickets containing multiple distinct requests, ensuring complex tickets are handled appropriately.

### Corpus Source Citation
Every LLM-generated response includes the exact corpus URL and retrieval confidence score in the justification field, making all decisions fully traceable.

---

## Escalation Logic
The agent escalates when:
- Fraud, identity theft, or account compromise is detected
- Account access loss or unauthorized access
- Security vulnerabilities or bug bounty reports
- Billing disputes with specific order IDs
- Legal threats or compliance concerns
- Prompt injection or adversarial manipulation detected
- Subscription cancellation or pausing requests
- Corpus does not contain sufficient information to answer safely

---

## Hackathon
Built as a submission for the **HackerRank Agentic AI Hackathon**.
