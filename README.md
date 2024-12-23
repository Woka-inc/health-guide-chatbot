# MySQL 데이터베이스 생성

#### 1. 데이터베이스 생성
```
CREATE DATABASE IF NOT EXISTS hgcb_db;
USE hgcb_db;
```

#### 2. 테이블 생성
user 테이블
```
CREATE TABLE user (
    user_id INT AUTO_INCREMENT PRIMARY KEY, 
    username VARCHAR(50) NOT NULL, 
    email VARCHAR(100) NOT NULL UNIQUE, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
    last_login TIMESTAMP NULL 
);
```

chat_title 테이블
```
CREATE TABLE chat_title (
    session_id INT AUTO_INCREMENT PRIMARY KEY, 
    user_id INT NOT NULL, 
    title VARCHAR(255) NOT NULL, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
    FOREIGN KEY (user_id) REFERENCES user(user_id) ON DELETE CASCADE 
);
```

chat_log 테이블
```
CREATE TABLE chat_log (
    chat_id INT AUTO_INCREMENT PRIMARY KEY, 
    session_id INT NOT NULL, 
    user_id INT NULL, 
    sender ENUM('user', 'ai') NOT NULL,
    message TEXT NOT NULL, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
    FOREIGN KEY (session_id) REFERENCES chat_title(session_id) ON DELETE CASCADE, 
    FOREIGN KEY (user_id) REFERENCES user(user_id) ON DELETE SET NULL 
);
```

### 3. database/config.py 파일 수정
line 6에 본인 MySQL password 입력