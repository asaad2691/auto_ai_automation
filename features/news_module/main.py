import feedparser

def fetch_news():
    """
    BBC RSS feed se latest 5 news headlines fetch karo
    """
    url = "http://feeds.bbci.co.uk/news/rss.xml"
    feed = feedparser.parse(url)

    if not feed.entries:
        return ["No news found at the moment."]

    headlines = []
    for entry in feed.entries[:5]:
        headlines.append(f"- {entry.title}")

    return headlines


def run(query: str = "") -> str:
    """
    Run function jo AI agent ke liye entrypoint hoga.
    """
    news_list = fetch_news()
    return "📰 Latest News:\n" + "\n".join(news_list)


if __name__ == "__main__":
    print(run())
