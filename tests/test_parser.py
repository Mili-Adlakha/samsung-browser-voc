"""Tests for Play Store review parser (formats A, B, C)."""

import hashlib

from services.parser import detect_format, parse_play_store_text

APP_VERSION = "30.XX"

FORMAT_A_SAMPLE = """\
John Doe
★★★★★
Great browser, fast and clean. Ad blocking works perfectly.
3 people found this review helpful
Did you find this helpful? Yes No

Jane Smith
★
Latest update broke everything. Lost all my tabs after sync.
74 people found this review helpful
Did you find this helpful? Yes No
"""

# Single newlines only (no blank line between reviews) — common Play Store paste
# Play Store web UI paste (author, date, text — no ★ stars)
FORMAT_PLAY_STORE_WEB = """\
Seth Weekley
25 May 2026
Not Google Chrome.
Did you find this helpful?

chi
25 May 2026
update sucks, completely changed the layout that ive been using for years. there were no useful new upgrades either, they just wanted to mess it all up. and the app didnt even ask me if i wanted to update or not. i cant change it back either.
Did you find this helpful?

Hoormazd Hoseini
25 May 2026
after recent update app is crashing continuously and keyboard freez a long time, please give update
Did you find this helpful?
"""

FORMAT_A_SINGLE_NEWLINE = """\
John Doe
★★★★★
Great browser, fast and clean. Ad blocking works perfectly.
3 people found this review helpful
Did you find this helpful? Yes No
Jane Smith
★
Latest update broke everything. Lost all my tabs after sync.
74 people found this review helpful
Did you find this helpful? Yes No
"""

FORMAT_B_SAMPLE = """\
reviewId,authorName,reviewText,starRating,thumbsUpCount,reviewCreatedVersion,at
abc123,John Doe,"Great browser, fast and clean",5,3,30.0.2.30,2026-05-22 10:30:00
def456,Jane Smith,Lost all tabs after update sync failure,1,74,30.0.0.63,2026-05-23 14:15:00
"""

FORMAT_B_ALT_HEADERS = """\
Review Text,Star Rating,Reviewer Name,Thumbs Up
Great browser fast and clean experience,5,John Doe,3
Lost all tabs after update sync failure here,1,Jane Smith,74
"""

FORMAT_C_SAMPLE = """\
1. Rating: 5/5 | John Doe | 22 May 2026
Great browser, fast and clean.
Upvotes: 3

2. Rating: 1/5 | Jane Smith | 23 May 2026
Lost all tabs after update. Terrible sync.
Upvotes: 74
"""

MIXED_QUALITY_SAMPLE = """\
John Doe
★★★★★
Great browser, fast and clean. Ad blocking works perfectly.
3 people found this review helpful

Short User
★★★★★
Too short
0 people found this review helpful

Jane Smith
★
Latest update broke everything. Lost all my tabs after sync.
74 people found this review helpful
"""

SINGLE_REVIEW_A = """\
Alex Kim
★★★
Solid browser overall but tabs feel sluggish sometimes.
12 people found this review helpful
"""


class TestDetectFormat:
    def test_detect_csv(self):
        assert detect_format(FORMAT_B_SAMPLE) == "csv"

    def test_detect_numbered(self):
        assert detect_format(FORMAT_C_SAMPLE) == "numbered"

    def test_detect_playstore_default(self):
        assert detect_format(FORMAT_A_SAMPLE) == "playstore"

    def test_plain_text_with_reviewid_word_not_csv(self):
        text = "The reviewId field was mentioned in a long forum post.\n" + ("x" * 20)
        assert detect_format(text) == "playstore"


class TestFormatPlayStoreWeb:
    def test_parses_author_date_text_without_stars(self):
        records = parse_play_store_text(FORMAT_PLAY_STORE_WEB, APP_VERSION)
        assert len(records) == 3
        assert records[0].author_name == "Seth Weekley"
        assert records[0].review_date == "25 May 2026"
        assert records[0].rating == 0
        assert "Not Google Chrome" in records[0].review_text
        assert records[1].author_name == "chi"
        assert "layout" in records[1].review_text
        assert records[2].author_name == "Hoormazd Hoseini"


class TestFormatA:
    def test_parses_two_reviews(self):
        records = parse_play_store_text(FORMAT_A_SAMPLE, APP_VERSION)
        assert len(records) == 2

    def test_parses_single_newline_paste(self):
        records = parse_play_store_text(FORMAT_A_SINGLE_NEWLINE, APP_VERSION)
        assert len(records) == 2

    def test_first_review_fields(self):
        records = parse_play_store_text(FORMAT_A_SAMPLE, APP_VERSION)
        first = records[0]
        assert first.author_name == "John Doe"
        assert first.rating == 5
        assert first.thumbs_up_count == 3
        assert "Ad blocking works perfectly" in first.review_text
        assert first.app_version == APP_VERSION

    def test_second_review_upvotes_and_rating(self):
        records = parse_play_store_text(FORMAT_A_SAMPLE, APP_VERSION)
        second = records[1]
        assert second.author_name == "Jane Smith"
        assert second.rating == 1
        assert second.thumbs_up_count == 74
        assert "Lost all my tabs" in second.review_text

    def test_review_id_format(self):
        records = parse_play_store_text(FORMAT_A_SAMPLE, APP_VERSION)
        text = records[0].review_text
        expected_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        assert records[0].review_id == f"v{APP_VERSION}_{expected_hash}_0"

    def test_single_review(self):
        records = parse_play_store_text(SINGLE_REVIEW_A, APP_VERSION)
        assert len(records) == 1
        assert records[0].rating == 3
        assert records[0].thumbs_up_count == 12


class TestFormatB:
    def test_parses_csv_rows(self):
        records = parse_play_store_text(FORMAT_B_SAMPLE, APP_VERSION)
        assert len(records) == 2

    def test_csv_alternate_headers(self):
        records = parse_play_store_text(FORMAT_B_ALT_HEADERS, APP_VERSION)
        assert len(records) == 2
        assert records[0].rating == 5

    def test_csv_metadata(self):
        records = parse_play_store_text(FORMAT_B_SAMPLE, APP_VERSION)
        assert records[0].author_name == "John Doe"
        assert records[0].rating == 5
        assert records[0].thumbs_up_count == 3
        assert records[0].review_date == "2026-05-22 10:30:00"
        assert records[1].thumbs_up_count == 74


class TestFormatC:
    def test_parses_numbered_blocks(self):
        records = parse_play_store_text(FORMAT_C_SAMPLE, APP_VERSION)
        assert len(records) == 2

    def test_numbered_fields(self):
        records = parse_play_store_text(FORMAT_C_SAMPLE, APP_VERSION)
        assert records[0].author_name == "John Doe"
        assert records[0].rating == 5
        assert records[0].thumbs_up_count == 3
        assert records[0].review_date == "22 May 2026"
        assert records[1].rating == 1
        assert records[1].thumbs_up_count == 74


class TestEdgeCases:
    def test_empty_string_returns_empty_list(self):
        assert parse_play_store_text("", APP_VERSION) == []
        assert parse_play_store_text("   \n  ", APP_VERSION) == []

    def test_mixed_quality_skips_short_reviews(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            records = parse_play_store_text(MIXED_QUALITY_SAMPLE, APP_VERSION)
        assert len(records) == 2
        authors = {r.author_name for r in records}
        assert "Short User" not in authors
        assert "John Doe" in authors
        assert "Jane Smith" in authors

    def test_minimum_body_length_enforced(self):
        short_only = """\
Bob
★★★★★
Too short here
"""
        assert parse_play_store_text(short_only, APP_VERSION) == []

    def test_ingested_at_is_iso(self):
        records = parse_play_store_text(SINGLE_REVIEW_A, APP_VERSION)
        assert records[0].ingested_at.endswith("+00:00") or "T" in records[0].ingested_at
