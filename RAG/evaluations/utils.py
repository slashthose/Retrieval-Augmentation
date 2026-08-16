import json
import csv


def load_queries(path):

    with open(path, "r", encoding="utf8") as f:

        return json.load(f)


def save_results(results, filename):

    with open(filename, "w", newline="", encoding="utf8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()

        writer.writerows(results)