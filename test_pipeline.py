import test_llm_trader
import test_ta

if __name__ == "__main__":
    symbol = "RELIANCE.NS"
    test_ta.main(symbol)
    test_llm_trader.main(symbol)