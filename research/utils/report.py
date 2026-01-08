import pandas as pd
import numpy as np
import io

class StrategyReporter:
    def __init__(self, initial_capital=1.0):
        self.initial_capital = initial_capital
    
    def calculate_metrics(self, equity_curve):
        """
        Calculate strategy performance metrics.
        equity_curve: Series of equity values.
        """
        if equity_curve.empty:
            return {}
            
        # 1. Total Return
        start_eq = self.initial_capital
        end_eq = equity_curve.iloc[-1]
        total_return = (end_eq - start_eq) / start_eq
        
        # 2. Max Drawdown
        running_max = equity_curve.cummax()
        drawdown = (equity_curve - running_max) / running_max
        max_dd = drawdown.min()
        
        # 3. Sharpe Ratio (Annualized)
        returns = equity_curve.pct_change().dropna()
        if returns.std() == 0:
            sharpe = 0.0
        else:
            sharpe = returns.mean() / returns.std() * np.sqrt(252) # Assuming daily
            
        return {
            "total_return": total_return,
            "max_drawdown": max_dd,
            "sharpe_ratio": sharpe
        }

    def generate_html(self, metrics, title="Strategy Report"):
        """
        Generate a simple HTML report string.
        """
        html = f"""
        <html>
        <head><title>{title}</title></head>
        <body>
            <h1>{title}</h1>
            <table border="1">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Return</td><td>{metrics.get('total_return', 0):.2%}</td></tr>
                <tr><td>Max Drawdown</td><td>{metrics.get('max_drawdown', 0):.2%}</td></tr>
                <tr><td>Sharpe Ratio</td><td>{metrics.get('sharpe_ratio', 0):.2f}</td></tr>
            </table>
        </body>
        </html>
        """
        return html
