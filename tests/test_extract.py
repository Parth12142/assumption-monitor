import pandas as pd
import pytest

from assumption_monitor import extract


def one(sentence):
    figures = extract.extract_figures(sentence)
    assert len(figures) == 1, figures
    return figures[0]


# ---------- The example from the plan ----------
def test_exchange_rate_example_from_the_plan():
    f = one("We assume the exchange rate will remain close to INR 83 per USD during the planning period.")
    assert (f["kind"], f["currency"], f["value"], f["unit"]) == ("money", "INR", 83.0, "per USD")
    assert f["limit_type"] == "approx" and f["metric_hint"] == "exchange rate"
    assert f["can_be_baseline"] and f["status"] == "ok"


# ---------- Numbers, abbreviations and magnitudes ----------
@pytest.mark.parametrize("sentence, expected", [
    ("Revenue was Rs. 42.5 crore last year.", 42.5e7),
    ("The budget is Rs 5 lakh in total.", 5e5),
    ("We raised USD 2 million in funding.", 2e6),
    ("Debt of $3 billion remains.", 3e9),
    ("It cost 2 million dollars to build.", 2e6),
    ("The fund holds ₹1,23,456 in cash.", 123456.0),
])
def test_money_is_normalized_to_plain_numbers(sentence, expected):
    f = one(sentence)
    assert f["kind"] == "money" and f["value"] == pytest.approx(expected)


@pytest.mark.parametrize("sentence", ["Margin was 10% in 2019.", "Margin was 10 percent in 2019.", "Margin was 10 per cent in 2019."])
def test_percent_can_be_written_three_ways(sentence):
    f = one(sentence)
    assert f["kind"] == "percent" and f["value"] == 10.0


def test_basis_points_and_percentage_points_are_kept_apart_from_percent():
    bps = one("Rates may rise by 50 basis points.")
    pp = one("Rates may rise by 1 percentage point.")
    assert (bps["kind"], bps["value"]) == ("percentage_points", 0.5)
    assert (pp["kind"], pp["value"]) == ("percentage_points", 1.0)


def test_signed_values():
    assert one("Prices fell -5% last year.")["value"] == -5.0
    assert one("Prices fell −5% last year.")["value"] == -5.0


def test_years_and_section_numbers_are_not_figures():
    assert extract.extract_figures("Section 7A of the 2019 report covers 3 topics.") == []


# ---------- Ranges ----------
def test_ranges_with_units_and_shared_units():
    f = one("We expect the crude oil price to stay between USD 65 and USD 80 per barrel.")
    assert (f["limit_type"], f["low"], f["high"], f["unit"]) == ("range", 65.0, 80.0, "per barrel")
    f = one("Prices will stay between 80 and 90 per barrel.")
    assert (f["kind"], f["low"], f["high"], f["unit"]) == ("number", 80.0, 90.0, "per barrel")
    f = one("Margins will land from 5% to 7%.")
    assert (f["low"], f["high"]) == (5.0, 7.0)
    f = one("Freight rates will rise 5-10% next year.")
    assert (f["low"], f["high"], f["period"]) == (5.0, 10.0, "next year")
    f = one("The loan is Rs. 5 to 7 crore.")
    assert (f["low"], f["high"]) == (5e7, 7e7)


# ---------- Limits ----------
@pytest.mark.parametrize("sentence, limit, inclusive", [
    ("Interest rates should stay below 8% this year.", "upper", False),
    ("Interest rates should stay under 8% this year.", "upper", False),
    ("Interest rates should stay at most 8% this year.", "upper", True),
    ("Interest rates should be no more than 8% this year.", "upper", True),
    ("Interest rates should be up to 8% this year.", "upper", True),
    ("Interest rates should stay above 8% this year.", "lower", False),
    ("Interest rates should be more than 8% this year.", "lower", False),
    ("Interest rates should be at least 8% this year.", "lower", True),
    ("Interest rates should be about 8% this year.", "approx", None),
])
def test_limit_words_give_a_direction(sentence, limit, inclusive):
    f = one(sentence)
    assert (f["limit_type"], f["inclusive"]) == (limit, inclusive)
    assert f["can_be_baseline"] is (limit == "approx")


# ---------- Meaning: price, total, growth, level ----------
def test_unit_price_total_growth_and_level_stay_distinct():
    assert one("Oil will average $80 per barrel.")["role"] == "unit_price"
    assert one("Revenue was Rs. 42.5 crore.")["role"] == "amount"
    growth = one("We expect demand growth of 8% per year.")
    assert (growth["role"], growth["change_direction"], growth["period"]) == ("change", "up", "per year")
    assert one("Sales will decline 5% next year.")["change_direction"] == "down"
    level = one("The interest rate on the loan is 10.5%.")
    assert level["role"] == "level" and level["can_be_baseline"]
    assert not growth["can_be_baseline"]


def test_a_word_after_the_number_can_show_a_change():
    f = one("Costs will be 3.5% higher next year.")
    assert (f["role"], f["change_direction"]) == ("change", "up")


def test_periods_are_found():
    assert one("Wages will rise 6% annually.")["period"] == "annually"
    assert one("We expect the interest rate to stay near 7% by 2028.")["period"] == "by 2028"


# ---------- Several figures and doubtful cases ----------
def test_each_figure_gets_its_own_metric_phrase():
    first, second = extract.extract_figures(
        "In 2025 the company earned revenue of Rs. 42.5 crore and an operating margin of 9.2%.")
    assert (first["metric_hint"], second["metric_hint"]) == ("revenue", "operating margin")
    assert first["status"] == second["status"] == "ok"


def test_a_number_without_a_metric_phrase_needs_a_human():
    f = one("The committee approved 12%.")
    assert f["status"] == "review" and "metric" in f["reason"]
    assert not f["can_be_baseline"]


def test_two_figures_for_the_same_metric_need_a_human():
    figures = extract.extract_figures("The interest rate is 8% now and the interest rate may be 9% later.")
    assert len(figures) == 2 and all(f["status"] == "review" for f in figures)


def test_negation_near_a_limit_needs_a_human():
    f = one("We do not expect interest rates to change by more than 1 percentage point.")
    assert f["limit_type"] == "lower" and f["status"] == "review" and "negated" in f["reason"]
    assert one("Interest rates should be no more than 8% this year.")["status"] == "ok"


# ---------- Files and command line ----------
def test_table_and_command_line(tmp_path, capsys):
    csv = tmp_path / "s.csv"
    pd.DataFrame({
        "sentence_id": ["a-1", "a-2", "a-3"],
        "sentence": ["We assume the exchange rate will stay near INR 83 per USD.", "The office is in Ohio.", "Interest rates will stay below 8% in 2026."],
    }).to_csv(csv, index=False)
    labels = tmp_path / "labels.csv"
    pd.DataFrame({"sentence_id": ["a-1", "a-3"], "is_assumption": ["1", "1"]}).to_csv(labels, index=False)
    out = tmp_path / "f.csv"
    assert extract.main(["--csv", str(csv), "--labels", str(labels), "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert table["sentence_id"].tolist() == ["a-1", "a-3"]
    assert "2 sentences, 2 figures" in capsys.readouterr().out


def test_bad_input_gives_clear_errors(tmp_path, capsys):
    assert extract.main([]) == 1
    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n1,2\n")
    assert extract.main(["--csv", str(bad)]) == 1
    assert "sentence_id" in capsys.readouterr().err
