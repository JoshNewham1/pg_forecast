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

CREATE FUNCTION arima_series(
    source_table TEXT,
    date_col TEXT,
    value_col TEXT
) RETURNS DOUBLE PRECISION[] AS $$
DECLARE arr DOUBLE PRECISION[];
BEGIN
    EXECUTE format(
        'SELECT array_agg(%I ORDER BY %I) FROM %I',
        value_col, date_col, source_table
    ) INTO arr;

    IF arr IS NULL OR array_length(arr,1) = 0 THEN
        RAISE EXCEPTION 'ARIMA: no data found in % for columns %, %',
            source_table, date_col, value_col;
    END IF;

    RETURN arr;
END;
$$ LANGUAGE plpgsql;

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

-- Used in unit tests to verify that a model has retrained
CREATE OR REPLACE FUNCTION model_get_id(
    model_type model,
    input_table TEXT,
    date_column TEXT,
    value_column TEXT
)
RETURNS BIGINT
SECURITY DEFINER
AS $$
DECLARE
    rec_result RECORD;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    EXECUTE format(
        'SELECT id
        FROM models
        WHERE model_type = %L AND input_table = %L AND date_column = %L AND value_column = %L
        LIMIT 1',
        
        model_type, input_table, date_column, value_column
    ) INTO rec_result;

    RETURN rec_result.id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION remove_forecast(
    model_type model,
    input_table TEXT,
    date_column TEXT,
    value_column TEXT
)
RETURNS BOOLEAN
SECURITY DEFINER
AS $$
DECLARE
    v_deleted_id BIGINT;
BEGIN
    -- Safety precaution for SECURITY DEFINER
    PERFORM set_config('search_path', 'public,pg_temp', true);

    EXECUTE format(
        'DELETE FROM models
        WHERE model_type = %L AND input_table = %L AND date_column = %L AND value_column = %L
        RETURNING id',
        
        model_type, input_table, date_column, value_column
    ) INTO v_deleted_id;

    IF v_deleted_id IS NOT NULL THEN
        RETURN TRUE;
    ELSE
        RETURN FALSE;
    END IF;
END;
$$ LANGUAGE plpgsql;