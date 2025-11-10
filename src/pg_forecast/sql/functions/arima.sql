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
CREATE OR REPLACE FUNCTION arima_css(
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
    -- Dynamically build AR and MA calculation strings
    ar_sum_sql TEXT := '0.0';
    ma_sum_sql TEXT := '0.0';
    lag_cols_sql TEXT := '';
    j INT;
    k INT;
    css_query TEXT;
    result_css DOUBLE PRECISION;
BEGIN
    -- ar_sum_sql: Dynamic number of windowed sums for AR terms
    FOR j IN 1..p LOOP
        lag_cols_sql := lag_cols_sql || format(
            ', LAG(%I, %s) OVER (ORDER BY %I) AS x_lag_%s',
            value_col, j, date_col, j
        );
        ar_sum_sql := ar_sum_sql || format(' + COALESCE(d.x_lag_%s * %L, 0.0)', j, phi_params[j]);
    END LOOP;

    -- ma_sum_sql: Dynamic number of summed residuals for MA terms
    FOR k IN 1..q LOOP
        ma_sum_sql := ma_sum_sql || format(' + COALESCE(r.past_errors[%s] * %L, 0.0)', k, theta_params[k]);
    END LOOP;

    css_query := format(
        $QUERY$
        WITH RECURSIVE
        -- Pre-calculate all AR lags (x_i-j)
        ordered_data AS (
            SELECT
                %I AS t,
                %I AS x_i,
                ROW_NUMBER() OVER (ORDER BY %I) AS i -- time index
                %s -- "lag_cols_sql" from above
            FROM %s
        ),

        -- Recursively calculate errors (a_i)
        -- a_i = x_i - (AR part) - (MA part) 
        error_calculator(i, x_i, ar_part, ma_part, a_i, past_errors) AS (
            -- Base Case: i = 1, no errors
            SELECT
                d.i,
                d.x_i,
                0.0::double precision AS ar_part,
                0.0::double precision AS ma_part,
                0.0::double precision AS a_i,
                ARRAY_FILL(0.0::double precision, ARRAY[%s]) AS past_errors -- q-length array of 0s
            FROM ordered_data d
            WHERE d.i = 1

            UNION ALL

            -- Recursive Step: i > 1
            SELECT
                d.i,
                d.x_i,
                -- AR: SUM(phi_j * x_i-j)
                (%s) AS ar_part,
                -- MA: SUM(theta_k * a_i-k)
                (%s) AS ma_part,
                -- Error: a_i = x_i - AR_part - MA_part
                (d.x_i - (%s) - (%s)) AS a_i,
                -- New history of errors: [new_a_i, old_a_i-1, ... ]
                ARRAY[(d.x_i - (%s) - (%s))] || r.past_errors[1:%s] AS past_errors
            FROM error_calculator r
            JOIN ordered_data d ON d.i = r.i + 1
        )
        -- CSS = SUM(a_i^2)
        SELECT sum(a_i * a_i)
        FROM error_calculator
        WHERE i > %s; -- Sum starts after p observations (otherwise inaccurate)
        $QUERY$,
        date_col, value_col, date_col, -- ordered_data SELECT
        lag_cols_sql,                  -- ordered_data lags
        source_table,                  -- ordered_data FROM
        q,                             -- Base case past_errors array_fill
        ar_sum_sql, ma_sum_sql,        -- Recursive ar_part, ma_part
        ar_sum_sql, ma_sum_sql,        -- Recursive a_i calculation
        ar_sum_sql, ma_sum_sql,        -- Recursive past_errors calculation
        q - 1,                         -- Slice for past_errors array
        p                              -- Final CSS SUM WHERE i > p
    );

    RAISE DEBUG 'Generated CSS Query: %', css_query;
    EXECUTE css_query INTO result_css;
    RETURN result_css;
END;
$$ LANGUAGE plpgsql;