"""Main entry point for Phase 1 trading pipeline."""
import json
from trading_llm.core.orchestrator import run_trading_cycle


def main():
    """Run trading cycle for multiple NSE symbols."""
    symbols = ["INFY.NS", "TCS.NS", "RELIANCE.NS"]
    
    print("Starting Phase 1 Trading Pipeline")
    print("=" * 50)
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        try:
            result = run_trading_cycle(symbol)
            
            # Print summary
            trade = result.get("trade", {})
            risk = result.get("risk", {})
            
            print(f"  Decision: {trade.get('decision', 'HOLD')}")
            print(f"  Confidence: {trade.get('confidence', 0.0):.2f}")
            print(f"  Risk Approved: {risk.get('approved', False)}")
            if not risk.get("approved", False):
                print(f"  Risk Reason: {risk.get('reason', '')}")
            
            # Print full result as JSON (compact)
            print(f"\n  Full result:")
            print(json.dumps(result, indent=2, default=str))
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
    
    print("\n" + "=" * 50)
    print("Pipeline complete. Check logs/trade_log.jsonl for full results.")


if __name__ == "__main__":
    main()

