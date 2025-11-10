-- Generic
CREATE OR REPLACE FUNCTION pg_forecast_train(
    model_name TEXT,
    table_name TEXT,
    date_col TEXT,
    value_col TEXT
)
RETURNS VOID
LANGUAGE plpgsql AS
$$
BEGIN
    -- TODO: Implement one-off model training?
    RETURN;
END;
$$;

CREATE OR REPLACE FUNCTION pg_forecast(
    model_name TEXT,
    input_table TEXT,
    date_col TEXT,
    value_col TEXT,
    horizon INT
)
RETURNS TABLE(date TIMESTAMP, forecast_value DOUBLE PRECISION) AS $$
BEGIN
    RETURN QUERY EXECUTE format(
        $f$
        SELECT
            CAST('1970-01-01' AS TIMESTAMP) AS date,
            CAST(1 AS DOUBLE PRECISION) AS forecast_value
        FROM generate_series(1, %s)
        $f$, horizon
    );
END;
$$ LANGUAGE plpgsql;
