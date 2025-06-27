-- schema.sql

-- Table: lcr_data
CREATE TABLE lcr_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    lcr_value NUMERIC NOT NULL
);

-- Table: nsfr_data
CREATE TABLE nsfr_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    nsfr_value NUMERIC NOT NULL
);

-- Table: irrbb_exposures
CREATE TABLE irrbb_exposures (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    portfolio TEXT NOT NULL,
    currency TEXT NOT NULL,
    maturity_bucket TEXT NOT NULL,
    amount NUMERIC NOT NULL,
    interest_rate NUMERIC NOT NULL
);

-- Table: rwa_data
CREATE TABLE rwa_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    exposure_class TEXT NOT NULL,
    ead NUMERIC NOT NULL,
    rwa NUMERIC NOT NULL
);
