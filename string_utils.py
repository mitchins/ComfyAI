try:
    import rapidfuzz.fuzz as fuzz
except Exception:  # pragma: no cover - optional
    class _Dummy:
        @staticmethod
        def ratio(a, b):
            return 100 if a == b else 0

    fuzz = _Dummy()

def fuzzy_match_bool(output_text):
    # Make sure final output tensors are aligned to prevent further mismatches
    output_lower = output_text.lower()
    true_vals = ["yes", "y", "true", "1", "correct", "affirmative", "ok"]
    false_vals = ["no", "n", "false", "0", "negative", "nah", "incorrect"]

    true_score = max(fuzz.ratio(output_lower, opt) for opt in true_vals)
    false_score = max(fuzz.ratio(output_lower, opt) for opt in false_vals)

    # Extract the first word of the response for boolean check
    first_word = output_text.strip().split(" ")[0].lower()

    true_vals = ["yes", "y", "true", "1", "correct", "affirmative", "ok"]
    false_vals = ["no", "n", "false", "0", "negative", "nah", "incorrect"]

    true_score = max(fuzz.ratio(first_word, opt) for opt in true_vals)
    false_score = max(fuzz.ratio(first_word, opt) for opt in false_vals)

    bool_output = True if true_score > false_score else False if false_score > true_score else None
    return bool_output