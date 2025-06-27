-- insert_dummy_data.sql

-- Data for lcr_data
INSERT INTO lcr_data (date, lcr_value) VALUES
('2024-06-27', 105),
('2024-07-31', 110),
('2024-08-31', 98);

-- Data for nsfr_data
INSERT INTO nsfr_data (date, nsfr_value) VALUES
('2024-06-27', 105),
('2024-07-31', 102),
('2024-08-31', 98);

-- Data for irrbb_exposures
INSERT INTO irrbb_exposures (date, portfolio, currency, maturity_bucket, amount, interest_rate) VALUES
('2024-06-27', 'Retail', 'EUR', '1-3 months', 1000000, 0.02),
('2024-06-27', 'Corporate', 'USD', '6-12 months', 500000, 0.025),
('2024-06-27', 'Retail', 'EUR', '3-5 years', 200000, 0.03);

-- Data for rwa_data
INSERT INTO rwa_data (date, exposure_class, ead, rwa) VALUES
('2024-06-27', 'Corporate', 500000, 400000),
('2024-06-27', 'Retail', 200000, 150000),
('2024-06-27', 'Sovereign', 1000000, 0);
