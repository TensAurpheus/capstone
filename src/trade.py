
class TradingStrategy:
    """
    Implement trading strategy based on model predictions
    """

    def __init__(self, model, threshold=0.5, position_size=1.0):
        self.model = model
        self.threshold = threshold
        self.position_size = position_size
        self.positions = []
        self.returns = []
        self.equity_curve = [1.0]  # Starting capital = 1

    def generate_signals(self, X_test, y_test_actual):
        """Generate trading signals from model predictions"""
        self.model.eval()
        signals = []

        with torch.no_grad():
            for i in range(len(X_test)):
                X_batch = torch.FloatTensor(X_test[i:i+1])
                pred = torch.sigmoid(self.model(X_batch)).item()

                # Generate signal: 1 for long, -1 for short, 0 for neutral
                if pred > self.threshold + 0.1:
                    signal = 1  # Long
                elif pred < self.threshold - 0.1:
                    signal = -1  # Short
                else:
                    signal = 0  # Neutral

                signals.append(signal)

        return np.array(signals)

    def backtest(self, signals, returns, transaction_cost=0.001):
        """Backtest the trading strategy"""
        equity = 1.0
        position = 0

        for i in range(len(signals)):
            # Check if we need to change position
            if signals[i] != position:
                # Close existing position
                if position != 0:
                    equity *= (1 - transaction_cost)

                # Open new position
                if signals[i] != 0:
                    equity *= (1 - transaction_cost)

                position = signals[i]

            # Calculate returns
            if position != 0:
                equity *= (1 + position * returns[i])

            self.equity_curve.append(equity)
            self.positions.append(position)

        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate performance metrics"""
        equity_array = np.array(self.equity_curve[1:])  # Exclude initial value
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]

        # Sharpe Ratio (assuming 252 trading days * 24 hours)
        sharpe = np.sqrt(252 * 24) * np.mean(returns) / np.std(returns)

        # Maximum Drawdown
        cummax = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cummax) / cummax
        max_drawdown = np.min(drawdown)

        # Total Return
        total_return = (equity_array[-1] - 1) * 100

        # Volatility
        volatility = np.std(returns) * np.sqrt(252 * 24)

        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'volatility': volatility,
            'final_equity': equity_array[-1]
        }
