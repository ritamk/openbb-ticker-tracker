"""Unit tests for parser and risk manager."""
import unittest
from trading_llm.core.utils import parse_json_response
from trading_llm.agents.risk_manager import evaluate_trade, clear_last_trade


class TestParseJsonResponse(unittest.TestCase):
    """Test JSON parsing with various edge cases."""
    
    def setUp(self):
        self.fallback = {"error": "parse_failed"}
    
    def test_valid_json(self):
        """Test parsing valid JSON."""
        text = '{"signal": "bullish", "confidence": 0.75}'
        result = parse_json_response(text, self.fallback)
        self.assertEqual(result["signal"], "bullish")
        self.assertEqual(result["confidence"], 0.75)
    
    def test_json_in_markdown(self):
        """Test parsing JSON from markdown code block."""
        text = '```json\n{"signal": "bearish", "confidence": 0.6}\n```'
        result = parse_json_response(text, self.fallback)
        self.assertEqual(result["signal"], "bearish")
        self.assertEqual(result["confidence"], 0.6)
    
    def test_json_in_code_block(self):
        """Test parsing JSON from generic code block."""
        text = '```\n{"signal": "neutral", "confidence": 0.5}\n```'
        result = parse_json_response(text, self.fallback)
        self.assertEqual(result["signal"], "neutral")
    
    def test_invalid_json(self):
        """Test fallback on invalid JSON."""
        text = 'not valid json {signal: bullish}'
        result = parse_json_response(text, self.fallback)
        self.assertEqual(result, self.fallback)
    
    def test_empty_text(self):
        """Test fallback on empty text."""
        result = parse_json_response("", self.fallback)
        self.assertEqual(result, self.fallback)
    
    def test_non_dict_json(self):
        """Test fallback on non-dict JSON."""
        text = '["list", "not", "dict"]'
        result = parse_json_response(text, self.fallback)
        self.assertEqual(result, self.fallback)


class TestRiskManager(unittest.TestCase):
    """Test risk manager gating logic."""
    
    def setUp(self):
        clear_last_trade()  # Clear state before each test
    
    def tearDown(self):
        clear_last_trade()  # Clean up after each test
    
    def test_atr_reject(self):
        """Test rejection when ATR% > 3%."""
        decision = {"decision": "BUY", "confidence": 0.8}
        ta_snapshot = {"atr_percent": 3.5, "realized_vol_20": 0.25}
        
        result = evaluate_trade("TEST.NS", decision, ta_snapshot)
        self.assertFalse(result["approved"])
        self.assertIn("ATR%", result["reason"])
    
    def test_atr_approve(self):
        """Test approval when ATR% <= 3%."""
        decision = {"decision": "BUY", "confidence": 0.8}
        ta_snapshot = {"atr_percent": 2.5, "realized_vol_20": 0.22}
        
        result = evaluate_trade("TEST.NS", decision, ta_snapshot)
        self.assertTrue(result["approved"])
        self.assertEqual(result["reason"], "Within limits")
    
    def test_duplicate_reject(self):
        """Test rejection of duplicate consecutive decisions."""
        decision1 = {"decision": "BUY", "confidence": 0.8}
        ta_snapshot = {"atr_percent": 2.0, "realized_vol_20": 0.20}
        
        # First decision should pass
        result1 = evaluate_trade("TEST.NS", decision1, ta_snapshot)
        self.assertTrue(result1["approved"])
        
        # Second identical decision should be rejected
        result2 = evaluate_trade("TEST.NS", decision1, ta_snapshot)
        self.assertFalse(result2["approved"])
        self.assertIn("Duplicate", result2["reason"])
    
    def test_different_decisions_approved(self):
        """Test that different decisions are both approved."""
        decision1 = {"decision": "BUY", "confidence": 0.8}
        decision2 = {"decision": "SELL", "confidence": 0.7}
        ta_snapshot = {"atr_percent": 2.0, "realized_vol_20": 0.20}
        
        result1 = evaluate_trade("TEST.NS", decision1, ta_snapshot)
        self.assertTrue(result1["approved"])
        
        result2 = evaluate_trade("TEST.NS", decision2, ta_snapshot)
        self.assertTrue(result2["approved"])
    
    def test_realized_vol_logged(self):
        """Test that realized volatility is always logged."""
        decision = {"decision": "HOLD", "confidence": 0.5}
        ta_snapshot = {"atr_percent": 2.0, "realized_vol_20": 0.28}
        
        result = evaluate_trade("TEST.NS", decision, ta_snapshot)
        self.assertIn("realized_vol_20", result)
        self.assertEqual(result["realized_vol_20"], 0.28)
    
    def test_symbol_isolation(self):
        """Test that duplicate checks are per-symbol."""
        decision = {"decision": "BUY", "confidence": 0.8}
        ta_snapshot = {"atr_percent": 2.0, "realized_vol_20": 0.20}
        
        # First decision for symbol1
        result1 = evaluate_trade("SYMBOL1.NS", decision, ta_snapshot)
        self.assertTrue(result1["approved"])
        
        # Same decision for symbol2 should pass (different symbol)
        result2 = evaluate_trade("SYMBOL2.NS", decision, ta_snapshot)
        self.assertTrue(result2["approved"])
        
        # Duplicate for symbol1 should fail
        result3 = evaluate_trade("SYMBOL1.NS", decision, ta_snapshot)
        self.assertFalse(result3["approved"])


if __name__ == "__main__":
    unittest.main()

