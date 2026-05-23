-- ===============================
-- Basel III Risk Database Schema
-- ===============================

-- ===============================
-- SCENARIOS TABLE
-- ===============================
CREATE TABLE scenarios (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    description TEXT,
    liquidity_shock NUMERIC(5,2) DEFAULT 0,  -- % withdrawal from liabilities
    ir_shift NUMERIC(5,2) DEFAULT 0,         -- Interest rate shift in basis points
    credit_shock NUMERIC(5,2) DEFAULT 0,     -- Credit spread or risk weight shock in basis points
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===============================
-- BALANCE SHEET TABLE
-- ===============================
CREATE TABLE balance_sheet (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    item VARCHAR(50) NOT NULL,         -- e.g., CET1, Tier1, Total Capital, Assets
    amount NUMERIC(18,2) NOT NULL,
    scenario_id INTEGER REFERENCES scenarios(id) ON DELETE SET NULL
);

-- ===============================
-- CASHFLOWS TABLE (LCR & NSFR)
-- ===============================
CREATE TABLE cashflows (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    product VARCHAR(50) NOT NULL,       -- e.g., loan, deposit, bond
    counterparty VARCHAR(50) NOT NULL,  -- retail, wholesale, interbank
    maturity_date DATE,
    bucket VARCHAR(20),                 -- e.g., 7d, 30d, 90d
    amount NUMERIC(18,2) NOT NULL,
    direction VARCHAR(10) CHECK (direction IN ('inflow', 'outflow')) NOT NULL,
    hqlatype VARCHAR(20) CHECK (hqlatype IN ('Level1', 'Level2A', 'Level2B', 'None')) NOT NULL,
    asf_factor NUMERIC(5,2) DEFAULT 0,  -- Available Stable Funding factor (NSFR)
    rsf_factor NUMERIC(5,2) DEFAULT 0,  -- Required Stable Funding factor (NSFR)
    scenario_id INTEGER REFERENCES scenarios(id) ON DELETE SET NULL
);

-- ===============================
-- RWA TABLE
-- ===============================
CREATE TABLE rwa (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    exposure_id VARCHAR(50) NOT NULL,
    asset_class VARCHAR(50) NOT NULL,       
    approach VARCHAR(20) CHECK (approach IN ('STD', 'IRB')) NOT NULL,
    amount NUMERIC(18,2) NOT NULL,           
    risk_weight NUMERIC(5,2) NOT NULL,       
    rwa_amount NUMERIC(18,2) NOT NULL,       -- Inserted by app: amount * risk_weight
    capital_requirement NUMERIC(18,2) NOT NULL, -- Inserted by app: rwa_amount * 0.08
    scenario_id INTEGER REFERENCES scenarios(id) ON DELETE SET NULL
);

-- ===============================
-- IRRBB TABLE
-- ===============================
CREATE TABLE irrbb (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    instrument VARCHAR(50) NOT NULL,     -- bond, loan, deposit, derivative
    cashflow NUMERIC(18,2) NOT NULL,     -- Amount of cashflow
    maturity_date DATE NOT NULL,         -- Maturity of cashflow
    tenor_bucket VARCHAR(20),            -- e.g., 0-1y, 1-3y, etc.
    pv01 NUMERIC(10,6) NOT NULL,         -- PV01 for this instrument
    rate_sensitivity NUMERIC(10,6),      -- Delta cashflow per 1bp shift
    scenario_id INTEGER REFERENCES scenarios(id) ON DELETE SET NULL
);

-- ===============================
-- PARAMS TABLE (Optional Config)
-- ===============================
CREATE TABLE params (
    key VARCHAR(50) PRIMARY KEY,
    value VARCHAR(100) NOT NULL
);
