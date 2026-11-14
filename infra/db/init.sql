CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE users (
  id          TEXT PRIMARY KEY,
  handle      TEXT UNIQUE NOT NULL,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TYPE visibility AS ENUM ('private', 'unlisted', 'public');

CREATE TABLE memories (
  id                TEXT PRIMARY KEY,
  owner_id          TEXT NOT NULL REFERENCES users(id),
  title             TEXT,
  description       TEXT,
  visibility        visibility NOT NULL DEFAULT 'private',
  location          geography(Point, 4326),
  captured_at       TIMESTAMPTZ,
  status            TEXT NOT NULL DEFAULT 'draft',
  manifest_version  INT NOT NULL DEFAULT 0,
  thumbnail_url     TEXT,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX memories_location_gix ON memories USING GIST (location);

CREATE TABLE memory_assets (
  memory_id  TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  version    INT NOT NULL,
  kind       TEXT NOT NULL,
  url        TEXT NOT NULL,
  bytes      BIGINT,
  checksum   TEXT,
  PRIMARY KEY (memory_id, version, kind, url)
);

CREATE TABLE jobs (
  id           TEXT PRIMARY KEY,
  memory_id    TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  stage        TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'queued',
  attempts     INT NOT NULL DEFAULT 0,
  gpu_seconds  DOUBLE PRECISION,
  error        TEXT,
  started_at   TIMESTAMPTZ,
  finished_at  TIMESTAMPTZ
);

CREATE TABLE reports (
  id           TEXT PRIMARY KEY,
  memory_id    TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  reporter_id  TEXT NOT NULL REFERENCES users(id),
  reason       TEXT NOT NULL,
  status       TEXT NOT NULL DEFAULT 'open',
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
