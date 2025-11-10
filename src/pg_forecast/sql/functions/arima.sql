CREATE OR REPLACE FUNCTION arima(
    d INT, -- Number of times to difference
    p INT, -- Number of lagged y_t
    q INT,  -- Number of lagged residuals
    horizon INT
)
RETURNS TABLE(date TIMESTAMP, forecast_value DOUBLE PRECISION) AS $$
BEGIN
    
END;
$$ LANGUAGE plpgsql;