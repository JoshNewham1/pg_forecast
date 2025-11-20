CREATE OR REPLACE FUNCTION arima(
    p INT, -- Number of lagged y_t
    d INT, -- Number of times to difference
    q INT,  -- Number of lagged residuals
    horizon INT
)
RETURNS TABLE(date TIMESTAMP, forecast_value DOUBLE PRECISION) AS $$
BEGIN
    
END;
$$ LANGUAGE plpgsql;

/**
 * Calculates the Conditional Sum of Squares (CSS) for an ARIMA(p,0,q) model
 * using a recursive CTE, based on the formula from Rosenthal & Lehner 2011.
 *
 * This function is intended for demonstration or for use with small data sets.
 *
 * @param source_table The properly-escaped name of the source data table.
 * @param date_col The name of the timestamp column (used for ordering).
 * @param value_col The name of the time series value column.
 * @param p The AR order
 * @param q The MA order
 * @param phi_params The array of AR parameters (phi_1, phi_2, ... phi_p).
 * @param theta_params The array of MA parameters (theta_1, theta_2, ... theta_q).
 * @returns The total Conditional Sum of Squares (CSS).
 */
CREATE OR REPLACE FUNCTION css_loss_sql(
    source_table TEXT,
    date_col TEXT,
    value_col TEXT,
    p INT,
    q INT,
    phi_params DOUBLE PRECISION[],
    theta_params DOUBLE PRECISION[]
)
RETURNS DOUBLE PRECISION AS $$
DECLARE
    ar_sum_sql TEXT := '0.0';
    ma_sum_sql TEXT := '0.0';
    lag_cols_sql TEXT := '';
    j INT;
    k INT;
    css_query TEXT;
    result_css DOUBLE PRECISION;
    ncond INT;
BEGIN
    ncond := GREATEST(p, q);

    -- Build AR lag expressions
    FOR j IN 1..p LOOP
        lag_cols_sql := lag_cols_sql || format(
            ', LAG(%I, %s) OVER (ORDER BY %I) AS x_lag_%s',
            value_col, j, date_col, j
        );
        ar_sum_sql := ar_sum_sql || format(' + COALESCE(d.x_lag_%s * %L, 0.0)', j, phi_params[j]);
    END LOOP;

    -- Build MA sum respecting min(i - ncond, q)
    ma_sum_sql := '0.0';
    FOR k IN 1..q LOOP
        ma_sum_sql := ma_sum_sql || format(
            ' + CASE WHEN r.i - %s >= %s THEN COALESCE(r.past_errors[%s] * %L, 0.0) ELSE 0.0 END',
            ncond, k, k, theta_params[k]
        );
    END LOOP;

    -- Construct recursive CSS query starting at i = ncond
    css_query := format(
        $QUERY$
        WITH RECURSIVE
        ordered_data AS (
            SELECT
                %I AS t,
                %I AS x_i,
                ROW_NUMBER() OVER (ORDER BY %I) AS i
                %s -- date_col, value_col, date_col, lag_cols_sql
            FROM %s
        ),

        -- Recursively calculate errors (a_i)
        -- a_i = x_i - (AR part) - (MA part) 
        error_calculator(i, x_i, ar_part, ma_part, a_i, past_errors) AS (
            -- Base Case: i = 1, no errors
            SELECT
                d.i,
                d.x_i,
                0.0::double precision AS a_i,
                ARRAY_FILL(0.0::double precision, ARRAY[%s]) AS past_errors
            FROM ordered_data d
            WHERE d.i = %s

            UNION ALL

            -- Recursive step: i > ncond
            SELECT
                d.i,
                d.x_i,
                (d.x_i - (%s) - (%s)) AS a_i,
                ARRAY[(d.x_i - (%s) - (%s))] || r.past_errors[1:%s] AS past_errors
            FROM error_calculator r
            JOIN ordered_data d ON d.i = r.i + 1
        )
        SELECT sum(a_i * a_i)
        FROM error_calculator;
        $QUERY$,
        date_col, value_col, date_col, lag_cols_sql, source_table, -- ordered_data
        q, ncond,                        -- base case past_errors length, start at i = ncond
        ar_sum_sql, ma_sum_sql,          -- residual a_i
        ar_sum_sql, ma_sum_sql,          -- past_errors array
        q                                -- slice past_errors
    );

    RAISE DEBUG 'Generated CSS Query: %', css_query;
    EXECUTE css_query INTO result_css;
    RETURN result_css;
END;
$$ LANGUAGE plpgsql;


CREATE FUNCTION css_loss(
    vals DOUBLE PRECISION[],
    phi DOUBLE PRECISION[],
    theta DOUBLE PRECISION[],
    p INT,
    q INT
)
RETURNS DOUBLE PRECISION
AS 'MODULE_PATHNAME', 'css_loss'
LANGUAGE C STRICT VOLATILE;

/*
SELECT
    css_loss(
        vals := array_agg(value ORDER BY t)::double precision[],
        phi := ARRAY[0.5]::double precision[],
        theta := ARRAY[0.3]::double precision[],
        p := 1,
        q := 1
    )
FROM
    time_series_data
WHERE
    series_id = 'TestSeries';
*/

CREATE FUNCTION optimise_arima(
    vals DOUBLE PRECISION[],
    p INT,
    q INT,
    method TEXT DEFAULT 'Nelder-Mead' -- 'Nelder-Mead', 'L-BFGS'
)
RETURNS DOUBLE PRECISION[]
AS 'MODULE_PATHNAME', 'optimise_arima'
LANGUAGE C STRICT VOLATILE;