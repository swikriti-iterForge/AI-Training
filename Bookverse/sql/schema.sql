-- Day 10 - hand-written schema + sample rows (practice database).
--   psql -U bookverse -d bookverse_practice -f sql/schema.sql

DROP TABLE IF EXISTS reviews, order_items, orders, books, authors, users CASCADE;

CREATE TABLE authors (
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(200) NOT NULL UNIQUE,
    bio   TEXT
);

CREATE TABLE books (
    id        SERIAL PRIMARY KEY,
    title     VARCHAR(300) NOT NULL,
    isbn      VARCHAR(13) UNIQUE,
    price     NUMERIC(10,2) NOT NULL CHECK (price > 0),
    stock     INTEGER NOT NULL DEFAULT 0 CHECK (stock >= 0),
    year      INTEGER,
    author_id INTEGER NOT NULL REFERENCES authors(id) ON DELETE RESTRICT
);
CREATE INDEX ix_books_title ON books(title);
CREATE INDEX ix_books_author_id ON books(author_id);

CREATE TABLE users (
    id              SERIAL PRIMARY KEY,
    email           VARCHAR(320) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL
);

CREATE TABLE orders (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    status     VARCHAR(20) NOT NULL DEFAULT 'confirmed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE order_items (
    id         SERIAL PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    book_id    INTEGER NOT NULL REFERENCES books(id) ON DELETE RESTRICT,
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10,2) NOT NULL
);

CREATE TABLE reviews (
    id      SERIAL PRIMARY KEY,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating  INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment TEXT,
    UNIQUE (book_id, user_id)
);

-- Sample rows (password hashes are fake placeholders)
INSERT INTO authors (name) VALUES ('George Orwell'), ('Frank Herbert'), ('James Clear'), ('Tara Westover');
INSERT INTO books (title, isbn, price, stock, year, author_id) VALUES
    ('1984', '9780451524935', 8.50, 15, 1949, 1),
    ('Animal Farm', NULL, 7.99, 11, 1945, 1),
    ('Dune', '9780441172719', 12.75, 3, 1965, 2),
    ('Atomic Habits', '9780735211292', 18.99, 12, 2018, 3),
    ('Educated', '9780399590504', 14.25, 0, 2018, 4);
INSERT INTO users (email, hashed_password) VALUES ('alice@example.com', 'x'), ('bob@example.com', 'x');
INSERT INTO orders (user_id) VALUES (1), (1);
INSERT INTO order_items (order_id, book_id, quantity, unit_price) VALUES (1, 3, 2, 12.75), (1, 1, 1, 8.50), (2, 4, 1, 18.99);
INSERT INTO reviews (book_id, user_id, rating, comment) VALUES (3, 1, 5, 'A classic.'), (3, 2, 4, NULL), (1, 1, 5, NULL);
