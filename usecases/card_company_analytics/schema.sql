-- Card company analytics: sessions, signup, merchant
-- Run this before data.sql. Safe to re-run (drops existing tables).

DROP TABLE IF EXISTS merchant;
DROP TABLE IF EXISTS signup;
DROP TABLE IF EXISTS sessions;

CREATE TABLE sessions (
    session_id          TEXT PRIMARY KEY,
    cookie_id           TEXT NOT NULL,
    start_timestamp      TIMESTAMPTZ NOT NULL,
    marketing_campaign_url TEXT
);

CREATE TABLE signup (
    signup_session_id   TEXT NOT NULL REFERENCES sessions(session_id),
    signup_date         DATE NOT NULL,
    account_id          TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE merchant (
    account_id          TEXT NOT NULL PRIMARY KEY REFERENCES signup(account_id),
    signup_date         DATE NOT NULL,
    kyc_submit_date     DATE,
    lifetime_volume     NUMERIC
);
