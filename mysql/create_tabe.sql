CREATE TABLE registered_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    face LONGTEXT,
    department VARCHAR(100)
);

CREATE TABLE logs_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    event_type VARCHAR(255),
    event_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    details TEXT
);

