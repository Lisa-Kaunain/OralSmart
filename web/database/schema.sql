-- Flask-SQLAlchemy creates these tables automatically on startup.
-- This reference schema is supplied for MySQL setup and project documentation.
CREATE DATABASE IF NOT EXISTS oralsmart_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE oralsmart_ai;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(120) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL,
    phone VARCHAR(40),
    created_at DATETIME NOT NULL
);

CREATE TABLE screenings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    image_filename VARCHAR(255) NOT NULL,
    ai_result VARCHAR(120) NOT NULL,
    confidence FLOAT NULL,
    explanation TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    model_mode VARCHAR(30) NOT NULL DEFAULT 'demo',
    review_status VARCHAR(30) NOT NULL DEFAULT 'Pending',
    review_requested BOOLEAN NOT NULL DEFAULT FALSE,
    created_at DATETIME NOT NULL,
    CONSTRAINT fk_screenings_patient FOREIGN KEY (patient_id) REFERENCES users(id),
    INDEX idx_screenings_patient (patient_id)
);

CREATE TABLE dentist_reviews (
    id INT AUTO_INCREMENT PRIMARY KEY,
    screening_id INT NOT NULL UNIQUE,
    dentist_id INT NOT NULL,
    professional_finding TEXT NOT NULL,
    treatment_plan TEXT NOT NULL,
    notes TEXT,
    follow_up_date DATE,
    review_status VARCHAR(30) NOT NULL,
    created_at DATETIME NOT NULL,
    CONSTRAINT fk_reviews_screening FOREIGN KEY (screening_id) REFERENCES screenings(id),
    CONSTRAINT fk_reviews_dentist FOREIGN KEY (dentist_id) REFERENCES users(id)
);
