# API reference

All API responses use JSON. Authentication is a Flask session cookie.

- `GET /api/health`: process, model availability, and database reachability.
- `POST /api/auth/signup`: JSON `name`, `email`, `password`.
- `POST /api/auth/login`: JSON `email`, `password`.
- `POST /api/auth/logout`: clears session.
- `GET /api/auth/me`: current user.
- `POST /api/analyze/image`: authenticated multipart upload under `image`.
- `GET /api/history`: current user's analyses.
- `GET /api/history/<analysis_id>`: one owned analysis.
- `DELETE /api/history/<analysis_id>`: delete one owned analysis.
- `POST /api/feedback`: authenticated JSON `category` and `message`.
- `GET /api/admin/summary`: admin-only database counts and model metadata.

Errors use `{ "success": false, "error": "Human-readable message" }`. Analysis is unavailable if the checkpoint or database is missing; no placeholder result is returned.
