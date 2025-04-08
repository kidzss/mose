import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from advisor.portfolio_advisor import PortfolioAdvisor

@pytest.fixture
async def mock_data_source():
    mock = AsyncMock()
    # Mock historical data
    mock.get_historical_data.return_value = pd.DataFrame({
        'date': pd.date_range(end=datetime.now(), periods=30),
        'close': [100 + i for i in range(30)],
        'volume': [1000000 for _ in range(30)]
    })
    return mock

@pytest.fixture
async def advisor(mock_data_source):
    return PortfolioAdvisor(data_source=mock_data_source)

@pytest.mark.asyncio
async def test_analyze_market_trend(advisor, mock_data_source):
    # Test parameters
    symbol = "AAPL"
    days = 30
    
    # Call the method
    result = await advisor.analyze_market_trend(symbol, days)
    
    # Verify the data source was called correctly
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    mock_data_source.get_historical_data.assert_called_once_with(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Verify result structure
    assert isinstance(result, dict)
    assert all(key in result for key in ['trend', 'confidence', 'volume_change', 'price_change'])
    
    # Verify value ranges
    assert -100 <= result['trend'] <= 100
    assert 0 <= result['confidence'] <= 100
    assert isinstance(result['volume_change'], float)
    assert isinstance(result['price_change'], float) 