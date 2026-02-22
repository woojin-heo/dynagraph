# Card company analytics

Use case for testing **sql_generation** and **sql_execution**: sessions, signups, and merchant KYC/volume data.

## Tables

| Table     | Description |
|----------|-------------|
| **sessions** | One row per session. `session_id` (PK), `cookie_id`, `start_timestamp`, `marketing_campaign_url`. |
| **signup**   | Only users who signed up. `signup_session_id` → `sessions.session_id`, `signup_date`, `account_id` (PK). |
| **merchant** | One row per account. `account_id` (PK), `signup_date`, `kyc_submit_date` (nullable), `lifetime_volume` (nullable). |

**Relations:** Signup links to session (campaign) via `signup_session_id = sessions.session_id`. Merchant links to signup via `account_id`. KYC submit = `merchant.kyc_submit_date IS NOT NULL`.

## Setup

1. Set `DATABASE_URL` and ensure PostgreSQL is running.
2. Run in order: `schema.sql` then `data.sql` (e.g. from `sql_test.ipynb` or `psql`).

When testing with the agent, `backend/db/tables.yaml` should describe these tables so the planner can choose `SQL_GENERATION`/ `SQL_EXECUTION`.

## Test questions

1. **Overall signup rate for 2024 (or 2025)**  
   Signup rate = (number of signups in the year) / (number of sessions in the year). Define year in the question.

2. **Rank marketing campaigns by KYC submit rate**  
   KYC submit rate = (count of KYC submits for that campaign) / (count of signups from that campaign). Join sessions → signup → merchant; filter `kyc_submit_date IS NOT NULL` for submits.

3. **Per monthly cohort, 5 campaigns with lowest KYC submission rate**  
   Cohort = month of signup (e.g. 2024-03). For each month, compute KYC submit rate per campaign, then return the 5 campaigns with the lowest rate (and optionally the rate).
