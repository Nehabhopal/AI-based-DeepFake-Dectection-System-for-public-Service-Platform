CREATE DATABASE IF NOT EXISTS deepsafe CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE deepsafe;

CREATE TABLE IF NOT EXISTS users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(120) NOT NULL,
  email VARCHAR(254) NOT NULL UNIQUE,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) NOT NULL DEFAULT 'user',
  created_at DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  image_path VARCHAR(500) NOT NULL,
  prediction VARCHAR(20) NOT NULL,
  confidence FLOAT NOT NULL,
  real_probability FLOAT NOT NULL,
  deepfake_probability FLOAT NOT NULL,
  image_quality VARCHAR(20) NOT NULL,
  face_detected BOOLEAN NOT NULL,
  model_version VARCHAR(40) NOT NULL,
  gradcam_path VARCHAR(500) NOT NULL,
  created_at DATETIME NOT NULL,
  INDEX idx_analyses_user_created (user_id, created_at),
  CONSTRAINT fk_analyses_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedback (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NULL,
  category VARCHAR(50) NOT NULL,
  message TEXT NOT NULL,
  created_at DATETIME NOT NULL,
  INDEX idx_feedback_created (created_at),
  CONSTRAINT fk_feedback_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);
