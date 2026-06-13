"""
generate_data.py
-----------------
Generates a synthetic labeled dataset of "tweets/reviews" for sentiment
analysis with three classes: positive, negative, neutral.

This is meant as SAMPLE DATA so the project runs end-to-end out of the box.
You can replace data/sentiment_data.csv with a real dataset (e.g. Sentiment140,
IMDB reviews, Amazon reviews) — just keep the two columns: "text" and "label",
where label is one of: positive, negative, neutral.
"""

import csv
import random

random.seed(42)

subjects = [
    "the product", "this movie", "the service", "this app", "the food",
    "the new phone", "this book", "the hotel room", "the laptop",
    "the customer support", "this game", "the delivery", "the show",
    "the restaurant", "this software", "the update", "my order",
    "the flight", "the course", "this album"
]

positive_templates = [
    "I absolutely love {s}, it exceeded all my expectations!",
    "{s} is amazing, highly recommend it to everyone.",
    "Wow, {s} works perfectly and looks great too.",
    "I'm so happy with {s}, best decision ever.",
    "{s} made my day, truly fantastic experience.",
    "What a great {s}, totally worth the money.",
    "I can't stop smiling, {s} is just wonderful.",
    "{s} is excellent, five stars without a doubt.",
    "Really impressed with {s}, will buy again.",
    "{s} was a pleasant surprise, loved every bit of it.",
    "Thank you so much, {s} is exactly what I needed.",
    "{s} is so good, I'm telling all my friends about it.",
    "Best experience ever with {s}, super satisfied!",
    "{s} is fast, reliable, and a joy to use.",
    "I'm thrilled with {s}, it works like a charm.",
]

negative_templates = [
    "I hate {s}, it was a complete waste of money.",
    "{s} is terrible and broke after one day.",
    "Worst experience ever, {s} did not work at all.",
    "I'm so disappointed with {s}, never buying again.",
    "{s} is awful, customer service ignored my complaints.",
    "Such a bad experience, {s} was completely useless.",
    "{s} arrived broken and nobody responded to my emails.",
    "I regret purchasing {s}, total waste of time.",
    "{s} is the worst thing I've ever used.",
    "Absolutely furious about {s}, this is unacceptable.",
    "{s} stopped working after a week, very frustrating.",
    "Do not buy {s}, it's a scam and poorly made.",
    "{s} was disgusting and overpriced, never again.",
    "I'm extremely unhappy with {s}, it ruined my day.",
    "{s} is buggy, slow, and frustrating to use.",
]

neutral_templates = [
    "{s} is okay, nothing special but does the job.",
    "I received {s} today, will update my review later.",
    "{s} arrived on time, packaging was standard.",
    "{s} is fine, similar to what I expected.",
    "Not sure how I feel about {s} yet.",
    "{s} works as described, no issues so far.",
    "{s} is average, could be better could be worse.",
    "I used {s} for a week, it's been fairly normal.",
    "{s} has both good and bad points, hard to say.",
    "{s} is just an ordinary product, nothing stands out.",
    "Here is some information about {s} for reference.",
    "{s} costs about the same as similar alternatives.",
    "{s} comes in different colors and sizes.",
    "{s} was delivered yesterday according to the tracking.",
    "I'm reading reviews about {s} before deciding.",
]

extra_positive = [
    "Loved it!", "Highly recommended!", "Five stars!", "So good!",
    "This is fantastic, thank you!", "Pure perfection.", "Made my week!",
]

extra_negative = [
    "Never again.", "Total disaster.", "Avoid at all costs.",
    "I want a refund.", "This is unacceptable.", "Such a letdown.",
]

extra_neutral = [
    "It is what it is.", "Will see how it goes.", "No strong opinion yet.",
    "Seems average to me.", "Just an FYI for others.", "Standard stuff.",
]


def build_rows():
    rows = []
    for s in subjects:
        for t in positive_templates:
            rows.append((t.format(s=s), "positive"))
        for t in negative_templates:
            rows.append((t.format(s=s), "negative"))
        for t in neutral_templates:
            rows.append((t.format(s=s), "neutral"))

    # add some short standalone examples
    for p in extra_positive:
        rows.append((p, "positive"))
    for n in extra_negative:
        rows.append((n, "negative"))
    for u in extra_neutral:
        rows.append((u, "neutral"))

    random.shuffle(rows)
    return rows


def main():
    rows = build_rows()
    out_path = "sentiment_data.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
