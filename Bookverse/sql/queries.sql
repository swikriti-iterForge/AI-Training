-- Day 10 - 10 practice queries.   psql -U bookverse -d bookverse_practice -f sql/queries.sql

-- 1. Books under $20 (filter)
SELECT title, price FROM books WHERE price < 20 ORDER BY price;
-- 2. Books with their author (JOIN)
SELECT b.title, a.name FROM books b JOIN authors a ON a.id = b.author_id ORDER BY a.name, b.title;
-- 3. Books per author, including authors with none (LEFT JOIN + aggregate)
SELECT a.name, COUNT(b.id) AS books FROM authors a LEFT JOIN books b ON b.author_id = a.id GROUP BY a.name ORDER BY books DESC;
-- 4. Out-of-stock books
SELECT title FROM books WHERE stock = 0;
-- 5. Case-insensitive title search
SELECT title FROM books WHERE title ILIKE '%dune%';
-- 6. Total of each order
SELECT o.id, SUM(oi.quantity * oi.unit_price) AS total FROM orders o JOIN order_items oi ON oi.order_id = o.id GROUP BY o.id ORDER BY o.id;
-- 7. Copies sold per book
SELECT b.title, SUM(oi.quantity) AS sold FROM order_items oi JOIN books b ON b.id = oi.book_id GROUP BY b.title ORDER BY sold DESC;
-- 8. Average rating per book (HAVING)
SELECT b.title, ROUND(AVG(r.rating), 2) AS avg_rating, COUNT(*) AS reviews FROM reviews r JOIN books b ON b.id = r.book_id GROUP BY b.title HAVING COUNT(*) >= 1;
-- 9. Users who never ordered
SELECT email FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);
-- 10. Stock decrease in a transaction with a row lock (what POST /orders does)
BEGIN;
SELECT id, stock FROM books WHERE id = 3 FOR UPDATE;
UPDATE books SET stock = stock - 1 WHERE id = 3 AND stock >= 1;
SELECT title, stock FROM books WHERE id = 3;
ROLLBACK;  -- practice only
