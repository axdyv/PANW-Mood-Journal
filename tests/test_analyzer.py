from src.analyzer import analyze_text

def test_analyze_text_returns_dict():
    result = analyze_text("I am crushing it at work!")
    assert isinstance(result, dict)
    assert "mood" in result
    assert "energy" in result
