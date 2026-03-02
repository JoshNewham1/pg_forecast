CREATE TYPE model AS ENUM ('autoarima', 'pyautoarima');

CREATE TABLE models(
    id BIGSERIAL PRIMARY KEY,
    model_type model NOT NULL,
    input_table TEXT NOT NULL,
    date_column TEXT NOT NULL,
    value_column TEXT NOT NULL,
    use_log_transform BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT models_unique_model UNIQUE (model_type, input_table, date_column, value_column)
);