import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).parents[1]


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if tag == "a" and "href" in values:
            self.links.append(values["href"])


def test_public_implementation_matches_recorded_hash() -> None:
    evidence = json.loads(
        (ROOT / "results/implementation_equivalence.json").read_text()
    )
    implementation = ROOT / evidence["public_implementation_path"]
    assert (
        hashlib.sha256(implementation.read_bytes()).hexdigest()
        == evidence["public_implementation_sha256"]
    )
    assert not (ROOT / "src/recirculation/author_recurrent_gemma3.py").exists()


def test_report_distinguishes_exploratory_and_confirmatory_runs() -> None:
    report = (ROOT / "report/index.html").read_text()
    required = [
        "current confirmatory run evaluates 40,920 predicted tokens",
        "13.503% reduction",
        "eight previously unseen books",
        "earlier exploratory run used 10 windows and 10,230 targets",
        "13.676%",
        "8 of 10 windows improved",
        "VIEW SOURCE / REPRODUCE ON GITHUB",
    ]
    assert all(text in report for text in required)
    assert "Move to eight unseen PG-19 books" not in report
    assert "This run evaluates 10,230" not in report


def test_report_public_links_are_release_ready() -> None:
    report = (ROOT / "report/index.html").read_text()
    parser = LinkParser()
    parser.feed(report)
    required_links = {
        "https://github.com/VargasDevelopment/recirculation-lab",
        "https://arxiv.org/abs/2608.17981",
        "https://gist.github.com/shoaibahmed/10702acc01cc5a169fdbc1719932438f",
        "https://github.com/VargasDevelopment/recirculation-lab/blob/main/results/validation_comparison.json",
        "https://github.com/VargasDevelopment/recirculation-lab/blob/main/results/comparison.json",
    }
    assert required_links <= set(parser.links)
    assert (
        'content="https://vargasdevelopment.github.io/recirculation-lab/og.png"'
        in report
    )


def test_result_artifacts_remain_distinct_and_complete() -> None:
    exploratory = json.loads((ROOT / "results/comparison.json").read_text())
    confirmatory = json.loads((ROOT / "results/validation_comparison.json").read_text())
    assert exploratory["evaluated_predicted_tokens"] == 10_230
    assert exploratory["windows"] == 10
    assert round(exploratory["percent_perplexity_reduction"], 3) == 13.676
    assert confirmatory["aggregate"]["evaluated_predicted_tokens"] == 40_920
    assert confirmatory["aggregate"]["windows"] == 40
    assert confirmatory["consistency"]["books_improved"] == 8
    assert round(confirmatory["aggregate"]["percent_perplexity_reduction"], 3) == 13.503
    assert len(confirmatory["per_window"]) == 40
