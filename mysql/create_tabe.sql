CREATE TABLE registered_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    face LONGTEXT,
    department VARCHAR(100)
);

CREATE TABLE logs_data (
    id INT NOT NULL AUTO_INCREMENT,
    name VARCHAR(255) NULL,
    face_dept VARCHAR(255) NULL,
    uniform_dept VARCHAR(255) NULL,
    status VARCHAR(100) NULL,
    captured_uuid VARCHAR(255) NULL,
    created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
);
