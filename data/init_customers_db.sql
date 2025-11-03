CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    address TEXT,
    orders 
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    item_name TEXT,
    price REAL,
    order_date TEXT,
    delivery_status TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

INSERT INTO users (id, name, email, address) VALUES
(1, 'Angela', 'angela@gmail.com', 'Angela Address');
INSERT INTO users (id, name, email, address) VALUES
(2, 'Evelyn', 'evelyn@gmail.com', 'Evelyn Address');
INSERT INTO users (id, name, email, address) VALUES
(3, 'Vivian', 'vivian@gmail.com', 'Vivian Address');

INSERT INTO orders (id, user_id, item_name, price, order_date, delivery_status) VALUES
(1, 1, 'Hand Soap', 5.99, '2025-03-25', 'delivered'),
(2, 1, 'Green Suitcase', 89.99, '2025-08-06', 'delivered'),
(3, 1, 'Brita Filter', 45.5, '2025-09-20', 'delivered'),
(4, 1, 'Blue Backpack', 38.7, '2025-10-24', 'shipped'),
(5, 2, 'Black Shoes', 35.2, '2025-10-25', 'pending'),
(6, 2, 'Black Sunglasses', 49.99, '2025-10-05', 'pending'),
(7, 2, 'Yellow Scarf', 29.99, '2025-07-12', 'delivered'),
(8, 2, 'Brown Belt', 24.99, '2025-09-28', 'shipped'),
(9, 3, 'Winter Gloves', 19.99, '2025-10-10', 'pending'),
(10, 3, 'Beanie Hat', 14.99, '2025-10-18', 'shipped'),
(11, 3, 'Green Jacket', 99.99, '2025-08-20', 'shipped'),
(12, 3, 'White Sneakers', 79.99, '2025-09-02', 'delivered');