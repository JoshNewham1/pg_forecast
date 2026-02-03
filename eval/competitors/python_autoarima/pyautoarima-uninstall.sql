BEGIN;

-- Drop triggers dynamically
DO $$
DECLARE r RECORD;
BEGIN
    FOR r IN
        SELECT trigger_name, event_object_table
        FROM information_schema.triggers
        WHERE trigger_name LIKE 'pyautoarima_on_insert_%'
    LOOP
        EXECUTE format(
            'DROP TRIGGER IF EXISTS %I ON %I;',
            r.trigger_name, r.event_object_table
        );
    END LOOP;
END;
$$;

-- Drop continuous aggregates
DO $$
DECLARE r RECORD;
BEGIN
    FOR r IN
        SELECT matviewname
        FROM pg_matviews
        WHERE matviewname LIKE 'pyautoarima_cagg_%'
    LOOP
        EXECUTE format('DROP MATERIALIZED VIEW IF EXISTS %I;', r.matviewname);
    END LOOP;
END;
$$;

-- Drop tables
DROP TABLE IF EXISTS pyautoarima_config;
DROP TABLE IF EXISTS pyautoarima_vertices;
DROP TABLE IF EXISTS model_pyautoarima_stats;

-- Drop functions
DROP FUNCTION IF EXISTS pyautoarima_on_insert_trigger();
DROP FUNCTION IF EXISTS pyautoarima_generate_vertices(BIGINT, INT, INT, INT, DOUBLE PRECISION[], DOUBLE PRECISION[], DOUBLE PRECISION, DOUBLE PRECISION);
DROP FUNCTION IF EXISTS pyautoarima_calc_state(DOUBLE PRECISION[],DOUBLE PRECISION[],DOUBLE PRECISION[],DOUBLE PRECISION,INT,INT,INT,DOUBLE PRECISION,DOUBLE PRECISION[],DOUBLE PRECISION[],DOUBLE PRECISION);
DROP FUNCTION IF EXISTS pyautoarima_init_state_helper(DOUBLE PRECISION[],DOUBLE PRECISION[],DOUBLE PRECISION[],DOUBLE PRECISION,INT,INT,INT);
DROP FUNCTION IF EXISTS pyautoarima_incremental_update(INT, INT, INT, pyautoarima_incremental_state, DOUBLE PRECISION);
DROP FUNCTION IF EXISTS pyautoarima_train(TEXT, TEXT, TEXT, BOOLEAN);
DROP FUNCTION IF EXISTS pyautoarima_fit_series(DOUBLE PRECISION[]);
DROP FUNCTION IF EXISTS pyautoarima_forecast(TEXT, TEXT, TEXT, INT, INTERVAL);

-- Drop types
DROP TYPE IF EXISTS pyautoarima_fit_result;
DROP TYPE IF EXISTS pyautoarima_incremental_state;

COMMIT;
