from backend.services import legal_parsing as lp


def test_durations_and_notice():
    n = lp.notice_periods("The Company may terminate by giving seven (7) days' notice. The Employee shall give ninety (90) days' notice.")
    assert sorted(x["days"] for x in n) == [7, 90]
    d = lp.extract_durations("a lock-in period of six (6) months")
    assert d and d[0]["days"] == 180


def test_indian_amounts():
    a = lp.extract_amounts("a deposit of Rs. 1,80,000 and rent of ₹18,000")
    assert [x["value"] for x in a] == [180000, 18000]
    assert lp.format_amount(180000) == "Rs. 1,80,000"


def test_clause_ref_prefers_printed_label():
    assert lp.clause_ref({"number": 4, "label": "3"}) == "3"
    assert lp.clause_ref({"number": 1, "label": None}) == "1"
