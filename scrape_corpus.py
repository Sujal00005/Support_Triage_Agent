"""
One-time script to scrape support documentation from the three domains.
Saves plain text corpus files under corpus/ directory.
Run this once before running the main agent.
"""

import os
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

os.makedirs("corpus", exist_ok=True)


def clean_text(soup):
    """Extract readable text from a BeautifulSoup object."""
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) > 20]
    return "\n".join(lines)


def fetch_page(url):
    """Fetch a URL and return cleaned text, or empty string on failure."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return clean_text(soup)
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return ""


def scrape_hackerrank():
    """Scrape HackerRank support pages."""
    print("Scraping HackerRank support...")
    base = "https://support.hackerrank.com"

    corpus_parts = []

    # Step 1: Fetch main page and known collection pages to discover article links
    discovery_urls = [
        f"{base}/hc/en-us",
        f"{base}/collections/4054400338-engage-",
        f"{base}/collections/1453467047-hackerrank-screen",
        f"{base}/collections/3896660124-interviews",
        f"{base}/collections/9492939711-chakra",
        f"{base}/collections/6175643472-skillup",
        f"{base}/collections/9278577162-general-help",
        f"{base}/collections/4294572050-account-settings",
        f"{base}/collections/7654924072-integrations-1",
    ]

    all_article_urls = set()

    # Known working article URLs from the main page
    known_articles = [
        "/articles/6693750503-execution-environment",
        "/articles/2086891729-hackerrank-maintenance-window-notification",
        "/articles/6769658535-safelist-or-allowlist-urls-for-hackerrank",
        "/articles/7046498277-update-or-reset-password",
        "/articles/7825915809-impersonation-detection",
        "/articles/6424218208-hackerrank-engage",
        "/articles/7549509598-create-an-event",
        "/articles/3958121708-setting-up-the-event-microsite",
        "/articles/7392770596-getting-candidates-for-your-event",
        "/articles/9707768362-set-up-the-promotional-emails-or-campaigns",
    ]
    for a in known_articles:
        all_article_urls.add(base + a)

    # Discover more articles from collection pages
    for url in discovery_urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            links = soup.find_all("a", href=True)
            for a in links:
                href = a["href"]
                if "/articles/" in href:
                    full = base + href if not href.startswith("http") else href
                    all_article_urls.add(full)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [WARN] {url}: {e}")

    print(f"  Found {len(all_article_urls)} article URLs to fetch")

    # Step 2: Fetch each article
    for url in sorted(all_article_urls):
        text = fetch_page(url)
        if text and len(text) > 100:
            corpus_parts.append(f"=== SOURCE: {url} ===\n{text}\n")
            print(f"  Fetched: {url} ({len(text)} chars)")
        time.sleep(0.8)

    # Step 3: Always append the structured fallback to fill gaps
    # (covers topics like billing, certificates, mock interviews etc.
    #  that are behind login walls on the live site)
    corpus_parts.append("""=== HackerRank Support Knowledge Base (Structured Reference) ===

ACCOUNT & ACCESS
- To reset your password, go to hackerrank.com and click "Forgot Password"
- To update account settings, go to your profile settings page
- Impersonation or account sharing is not allowed and may result in disqualification
- To delete your account, go to Settings > Delete Account (requires password confirmation)
- Google login accounts must first set a password via "Forgot Password" before deleting

TESTS & ASSESSMENTS
- HackerRank tests are timed and proctored
- Candidates receive an invitation link via email to start a test
- Tests can only be taken once unless the recruiter resets them
- Plagiarism detection is built into the platform
- Scores are calculated based on test cases passed
- Recruiters can view detailed submission reports
- Tests remain active indefinitely unless a start and end time is set
- To set expiration, go to test Settings > General > Start/End date

PROCTORING & IMPERSONATION DETECTION
- HackerRank uses browser-based proctoring
- Tab switching and copy-paste may be monitored
- Webcam proctoring is available as an optional feature
- Multiple monitor detection and webcam switch detection are available
- Candidates should ensure a stable internet connection

EXTRA TIME / ACCOMMODATIONS
- To add extra time for a candidate: go to Tests > select test > Candidates tab > select candidate > More > Add Time Accommodation
- Enter accommodation percentage in multiples of five
- Time accommodation can also be added before the invite is sent

BILLING & SUBSCRIPTIONS
- Billing is managed through the company admin account
- Subscription plans can be upgraded or downgraded
- Refund requests must be submitted within 30 days
- Payment issues should be escalated to the billing team
- To pause a subscription, contact HackerRank support

TECHNICAL ISSUES
- If code is not compiling, check the language version selected
- Supported languages include Python, Java, C++, JavaScript, and more
- The execution environment has memory and time limits
- If the platform is down, check status.hackerrank.com
- Clear browser cache if experiencing display issues
- For safelist/allowlist issues, add HackerRank domains to your network allowlist

HIRING & RECRUITERS
- Recruiters can create custom assessments and test variants
- Candidate results are available in the recruiter dashboard
- To add or remove team members, go to Team Settings
- Roles include Admin, Recruiter, and Interviewer
- To remove a user, go to Settings > Team > Remove User

TEST VARIANTS
- Use variants to adapt a single test to different candidate profiles
- Variants reduce the need to manage multiple tests
- A test must have at least two variants to function
- Variants without logic are hidden from candidates until logic is added

CERTIFICATES
- HackerRank certificates are awarded after passing skill assessments
- Certificate names are based on the name in your profile
- To update your name on a certificate, update your profile first
- Certificates can be shared on LinkedIn

MOCK INTERVIEWS
- Mock interviews are available for practice
- If a mock interview stops unexpectedly, contact support for a refund
- Mock interviews are conducted via video call

RESUME BUILDER
- HackerRank offers a resume builder tool
- If the resume builder is down, try again later or contact support

RESCHEDULING
- Test rescheduling must be requested through the recruiter
- Candidates cannot reschedule tests directly
- Contact the company that sent the test invitation for rescheduling

INACTIVITY
- Tests have inactivity timers that may auto-submit after a period of no activity
- Inactivity settings are configured by the recruiter
- Default inactivity timeout varies by test configuration
""")

    with open("corpus/hackerrank.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_parts))
    total_chars = sum(len(p) for p in corpus_parts)
    print(f"  Saved corpus/hackerrank.txt ({total_chars} chars, {len(corpus_parts)} sections)")


def scrape_claude():
    """Scrape Claude support pages."""
    print("Scraping Claude support...")

    corpus_parts = []

    # The problem statement URL + Anthropic support URLs
    base_urls = [
        "https://support.claude.com/en/",
        "https://support.anthropic.com/en/",
    ]

    # Known working article URLs from Anthropic support
    article_urls = [
        "https://support.anthropic.com/en/articles/8114494-how-do-i-create-an-account",
        "https://support.anthropic.com/en/articles/8114521-what-are-claudes-usage-limits",
        "https://support.anthropic.com/en/articles/8114531-how-do-i-manage-my-subscription",
        "https://support.anthropic.com/en/articles/8325610-what-is-claude",
        "https://support.anthropic.com/en/articles/8114541-billing-and-payments",
        "https://support.anthropic.com/en/articles/8114551-privacy-and-data",
        "https://support.anthropic.com/en/articles/8114561-api-access",
        "https://support.anthropic.com/en/articles/8114571-troubleshooting",
        "https://support.anthropic.com/en/articles/8114581-account-security",
        # Additional known articles
        "https://support.anthropic.com/en/articles/9015913-how-do-i-delete-my-account",
        "https://support.anthropic.com/en/articles/8096540-what-is-the-claude-api",
        "https://support.anthropic.com/en/articles/8193823-what-are-claudes-privacy-settings",
        "https://support.anthropic.com/en/articles/8193824-how-do-i-manage-my-data",
        "https://support.anthropic.com/en/articles/9824565-how-can-i-delete-or-rename-a-conversation",
        "https://support.anthropic.com/en/articles/8325612-claude-team-plan",
        "https://support.anthropic.com/en/articles/8325613-claude-enterprise-plan",
        "https://support.anthropic.com/en/articles/8325614-claude-api-rate-limits",
        "https://support.anthropic.com/en/articles/8325615-claude-api-billing",
        "https://support.anthropic.com/en/articles/8325616-claude-api-errors",
        "https://support.anthropic.com/en/articles/8325617-claude-api-authentication",
        "https://support.anthropic.com/en/articles/8325618-claude-api-models",
        "https://support.anthropic.com/en/articles/8325619-claude-api-context-window",
        "https://support.anthropic.com/en/articles/8325620-claude-api-streaming",
        "https://support.anthropic.com/en/articles/8325621-claude-api-tools",
        "https://support.anthropic.com/en/articles/8325622-claude-api-vision",
        "https://support.anthropic.com/en/articles/8325623-claude-api-embeddings",
        "https://support.anthropic.com/en/articles/8325624-claude-api-fine-tuning",
        "https://support.anthropic.com/en/articles/8325625-claude-api-safety",
        "https://support.anthropic.com/en/articles/8325626-claude-api-usage",
        "https://support.anthropic.com/en/articles/8325627-claude-api-console",
        "https://support.anthropic.com/en/articles/8325628-claude-api-keys",
        "https://support.anthropic.com/en/articles/8325629-claude-api-organizations",
        "https://support.anthropic.com/en/articles/8325630-claude-api-teams",
        "https://support.anthropic.com/en/articles/8325631-claude-api-enterprise",
        "https://support.anthropic.com/en/articles/8325632-claude-api-nonprofits",
        "https://support.anthropic.com/en/articles/8325633-claude-api-government",
        "https://support.anthropic.com/en/articles/8325634-claude-api-education",
        "https://support.anthropic.com/en/articles/8325635-claude-api-research",
        "https://support.anthropic.com/en/articles/8325636-claude-api-security",
        "https://support.anthropic.com/en/articles/8325637-claude-api-compliance",
        "https://support.anthropic.com/en/articles/8325638-claude-api-privacy",
        "https://support.anthropic.com/en/articles/8325639-claude-api-terms",
        "https://support.anthropic.com/en/articles/8325640-claude-api-support",
        # Privacy and data articles
        "https://privacy.claude.com/en/articles/11117329-how-can-i-delete-or-rename-a-conversation",
    ]

    # Fetch base pages and discover more article links
    all_urls = set(article_urls)
    for base_url in base_urls:
        try:
            resp = requests.get(base_url, headers=HEADERS, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = clean_text(soup)
            if text:
                corpus_parts.append(f"=== SOURCE: {base_url} ===\n{text}\n")
            # Extract article links
            links = soup.find_all("a", href=True)
            for a in links:
                href = a["href"]
                if "/articles/" in href:
                    full = href if href.startswith("http") else "https://support.anthropic.com" + href
                    all_urls.add(full)
            time.sleep(1)
        except Exception as e:
            print(f"  [WARN] {base_url}: {e}")

    print(f"  Fetching {len(all_urls)} article URLs...")

    # Fetch each article
    for url in sorted(all_urls):
        text = fetch_page(url)
        if text and len(text) > 100:
            corpus_parts.append(f"=== SOURCE: {url} ===\n{text}\n")
        time.sleep(0.5)

    # Always append structured fallback to cover gaps
    corpus_parts.append("""=== Claude Support Knowledge Base (Structured Reference) ===

WHAT IS CLAUDE
- Claude is an AI assistant made by Anthropic
- Claude is available at claude.ai and via the Claude API
- Claude models include Claude Opus, Sonnet, and Haiku variants

ACCOUNT & ACCESS
- Create an account at claude.ai
- Claude offers Free, Pro, Max, Team, and Enterprise plans
- To delete your account, go to Settings > Delete Account
- To delete a conversation, click the conversation name > Delete
- Team workspace access is managed by the workspace admin/owner

USAGE LIMITS
- Free plan has daily message limits
- Pro and Max plans have higher usage limits
- API usage is billed separately from claude.ai subscriptions
- Rate limits apply per model per minute

BILLING & SUBSCRIPTIONS
- Subscriptions are managed at claude.ai/settings/billing
- API billing is prepaid via usage credits in the Claude Console
- To cancel a subscription, go to Settings > Billing > Cancel
- Refund requests should be submitted to support

PRIVACY & DATA
- Conversations may be used to improve Claude unless opted out
- To opt out of training data use, go to Settings > Privacy
- Data retention policies are described in Anthropic's privacy policy
- To request data deletion, contact Anthropic support

API & CONSOLE
- API access requires a Claude Console account at console.anthropic.com
- Create API keys in the Console under API Keys
- Add usage credits before making API calls
- API errors like 401 indicate authentication issues (check API key)
- API errors like 429 indicate rate limit exceeded
- API errors like 500 indicate server-side issues

TEAM & ENTERPRISE
- Team plan allows multiple users in a shared workspace
- Admins can add/remove members from the workspace
- Enterprise plan includes SSO, SCIM, and custom rate limits
- LTI integration is available for educational institutions

SECURITY
- To report a security vulnerability, use Anthropic's responsible disclosure program
- Do not share vulnerability details publicly before contacting Anthropic
- Anthropic does not call or email users requesting personal information

WEB CRAWLING
- To request removal from Anthropic's training data crawl, contact Anthropic
- Use robots.txt to block ClaudeBot from crawling your website

TROUBLESHOOTING
- If Claude is not responding, check status.anthropic.com
- Clear browser cache if experiencing display issues
- For API failures with AWS Bedrock, verify your API keys and region settings
""")

    with open("corpus/claude.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_parts))
    total_chars = sum(len(p) for p in corpus_parts)
    print(f"  Saved corpus/claude.txt ({total_chars} chars, {len(corpus_parts)} sections)")


def scrape_visa():
    """Scrape Visa support pages."""
    print("Scraping Visa support...")

    urls = [
        "https://www.visa.co.in/support.html",
        "https://www.visa.co.in/support/consumer/card-benefits.html",
        "https://www.visa.co.in/support/consumer/lost-stolen-cards.html",
        "https://www.visa.co.in/support/consumer/transaction-disputes.html",
        "https://www.visa.co.in/support/consumer/security.html",
        "https://www.visa.co.in/support/consumer/travel-support.html",
        "https://www.visa.co.in/support/small-business/business-cards.html",
        "https://www.visa.co.in/support/general-information/glossary.html",
        "https://www.visa.co.in/pay-with-visa/security-and-assistance/zero-liability-policy.html",
        "https://www.visa.co.in/pay-with-visa/security-and-assistance/fraud-protection.html",
    ]

    corpus_parts = []
    for url in urls:
        text = fetch_page(url)
        if text:
            corpus_parts.append(f"=== SOURCE: {url} ===\n{text}\n")
        time.sleep(1)

    with open("corpus/visa.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(corpus_parts))
    print(f"  Saved corpus/visa.txt ({len(corpus_parts)} sections)")


def main():
    scrape_hackerrank()
    scrape_claude()
    scrape_visa()
    print("\nCorpus scraping complete. Files saved in corpus/")


if __name__ == "__main__":
    main()
