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

CREATE TYPE model AS ENUM ('autoarima');

CREATE TABLE models(
    id BIGSERIAL PRIMARY KEY,
    model_type model NOT NULL,
    input_table TEXT NOT NULL,
    date_column TEXT NOT NULL,
    value_column TEXT NOT NULL,
    horizon INT NOT NULL,
    UNIQUE (model_type, input_table, date_column, value_column)
);

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