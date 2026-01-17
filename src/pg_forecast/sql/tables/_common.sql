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