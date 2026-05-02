"""Re-scrape HackerRank and Claude corpus only."""
from scrape_corpus import scrape_hackerrank, scrape_claude
scrape_hackerrank()
scrape_claude()
print("\nDone. Run python agent.py next.")
