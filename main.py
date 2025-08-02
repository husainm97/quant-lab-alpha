from src.factor_model import run_factor_regression

def main():
    print("Starting Fama-French Factor Regression analysis...")
  
    # For now, call main regression function
    results = run_factor_regression()
    
    print(results.summary())

if __name__ == "__main__":
    main()
