CREATE TYPE kpss_result AS (
    kpss_val DOUBLE PRECISION,
    crit_val DOUBLE PRECISION,
    test_passed BOOLEAN
);

CREATE FUNCTION kpss(
    vals DOUBLE PRECISION[], -- Ordered array of time-series values
    p_val DOUBLE PRECISION DEFAULT 0.05 -- (0.1, 0.05, 0.025, 0.01)
)
RETURNS kpss_result
AS 'MODULE_PATHNAME', 'kpss'
LANGUAGE C STRICT STABLE;

CREATE FUNCTION aicc(
    css DOUBLE PRECISION,
    p INT,
    q INT,
    k INT,
    n_vals INT
)
RETURNS DOUBLE PRECISION
AS 'MODULE_PATHNAME', 'aicc'
LANGUAGE C STRICT STABLE;

CREATE OR REPLACE FUNCTION auto_arima(
    horizon INT,
    source_table TEXT, -- Table/view name
    date_col TEXT,     -- Timestamp column
    value_col TEXT     -- Numerical value column
)
RETURNS TABLE(date TIMESTAMP, forecast_value DOUBLE PRECISION) AS $$
DECLARE
    vals DOUBLE PRECISION[];
    diff_vals DOUBLE PRECISION[];
    d INT := 0;
    kpss_res kpss_result;
BEGIN
    -- Aggregate the series into an array
    EXECUTE format(
        'SELECT array_agg(%I ORDER BY %I) FROM %I',
        value_col,
        date_col,
        source_table
    )
    INTO vals;

    diff_vals := vals;
    FOR i IN 0..3 LOOP
        kpss_res := kpss(diff_vals);
        EXIT WHEN kpss_res.test_passed;
        d := d + 1;
        -- If test failed, difference and try again
        diff_vals := arima_difference(diff_vals, d);
    END LOOP;

    IF d > 2 THEN
        RAISE EXCEPTION 'More than 2 differences required, ARIMA model not suitable';
    END IF;

    RAISE NOTICE '% differences required', d;
END;
$$ LANGUAGE plpgsql;