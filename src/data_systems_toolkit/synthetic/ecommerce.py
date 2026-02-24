"""
E-commerce synthetic data generator - Shopify-style transactional data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from faker import Faker

from ..core.logging import get_logger

logger = get_logger(__name__)


class EcommerceDataGenerator:
 """Generate Shopify-style e-commerce synthetic data with referential integrity."""

 def __init__(self, random_seed: Optional[int] = None):
 """Initialize e-commerce data generator.

 Args:
 random_seed: Random seed for reproducibility
 """
 self.random_seed = random_seed or 42
 np.random.seed(self.random_seed)

 self.faker = Faker
 self.faker.seed_instance(self.random_seed)

 # Shared entity pools for referential integrity
 self._customers: Optional[pd.DataFrame] = None
 self._products: Optional[pd.DataFrame] = None
 self._orders: Optional[pd.DataFrame] = None

 logger.info(
 f"Initialized EcommerceDataGenerator with seed {self.random_seed}"
 )

 def generate_customers(self, num_customers: int = 200) -> pd.DataFrame:
 """Generate customer data.

 Args:
 num_customers: Number of customers to generate

 Returns:
 pd.DataFrame: Customer data
 """
 logger.info(f"Generating {num_customers} customers")

 data = []
 for i in range(num_customers):
 created_at = self.faker.date_time_between(
 start_date="-2y", end_date="-30d"
 )
 data.append(
 {
 "customer_id": f"cust_{i + 1:06d}",
 "email": self.faker.email,
 "first_name": self.faker.first_name,
 "last_name": self.faker.last_name,
 "phone": self.faker.phone_number,
 "created_at": created_at,
 "updated_at": self.faker.date_time_between(
 start_date=created_at, end_date="now"
 ),
 "accepts_marketing": np.random.choice(
 [True, False], p=[0.4, 0.6]
 ),
 "orders_count": 0, # Will be updated after orders generated
 "total_spent": 0.0, # Will be updated after orders generated
 "state": self.faker.state_abbr,
 "country": "US",
 "tags": np.random.choice(
 ["", "vip", "wholesale", "returning"],
 p=[0.6, 0.1, 0.1, 0.2],
 ),
 }
 )

 self._customers = pd.DataFrame(data)
 return self._customers.copy

 def generate_products(self, num_products: int = 50) -> pd.DataFrame:
 """Generate product catalog data.

 Args:
 num_products: Number of products to generate

 Returns:
 pd.DataFrame: Product data
 """
 logger.info(f"Generating {num_products} products")

 categories = [
 "Electronics",
 "Clothing",
 "Home & Garden",
 "Sports",
 "Beauty",
 "Books",
 ]

 # Price ranges by category (min, max, typical margin)
 price_config = {
 "Electronics": (50, 1500, 0.25),
 "Clothing": (20, 200, 0.50),
 "Home & Garden": (25, 500, 0.40),
 "Sports": (15, 300, 0.35),
 "Beauty": (10, 150, 0.60),
 "Books": (10, 50, 0.40),
 }

 data = []
 for i in range(num_products):
 category = np.random.choice(categories)
 min_price, max_price, _ = price_config[category]

 # Log-normal distribution for more realistic pricing
 price = np.exp(
 np.random.uniform(np.log(min_price), np.log(max_price))
 )
 price = round(price, 2)

 # Compare at price (original price before discount)
 has_discount = np.random.random < 0.3
 compare_at_price = (
 round(price * np.random.uniform(1.1, 1.5), 2)
 if has_discount
 else None
 )

 created_at = self.faker.date_time_between(
 start_date="-3y", end_date="-60d"
 )

 data.append(
 {
 "product_id": f"prod_{i + 1:05d}",
 "title": f"{self.faker.word.title} {category} Item",
 "vendor": self.faker.company,
 "product_type": category,
 "created_at": created_at,
 "updated_at": self.faker.date_time_between(
 start_date=created_at, end_date="now"
 ),
 "published_at": created_at + timedelta(days=np.random.randint(1, 7)),
 "status": np.random.choice(
 ["active", "draft", "archived"], p=[0.85, 0.10, 0.05]
 ),
 "price": price,
 "compare_at_price": compare_at_price,
 "sku": f"SKU-{category[:3].upper}-{i + 1:05d}",
 "inventory_quantity": np.random.randint(0, 500),
 "weight": round(np.random.uniform(0.1, 10.0), 2),
 "weight_unit": "kg",
 }
 )

 self._products = pd.DataFrame(data)
 return self._products.copy

 def generate_orders(
 self,
 num_orders: int = 1000,
 customers: Optional[pd.DataFrame] = None,
 products: Optional[pd.DataFrame] = None,
 ) -> pd.DataFrame:
 """Generate order data with line items.

 Args:
 num_orders: Number of orders to generate
 customers: Customer DataFrame (uses internal if not provided)
 products: Product DataFrame (uses internal if not provided)

 Returns:
 pd.DataFrame: Order data
 """
 customers = customers if customers is not None else self._customers
 products = products if products is not None else self._products

 if customers is None:
 customers = self.generate_customers
 if products is None:
 products = self.generate_products

 logger.info(f"Generating {num_orders} orders")

 # Active products only
 active_products = products[products["status"] == "active"]

 data = []
 for i in range(num_orders):
 # Select customer (some customers order more than others)
 customer_weights = np.random.dirichlet(np.ones(len(customers)) * 0.5)
 customer = customers.sample(weights=customer_weights).iloc[0]

 # Order timing
 created_at = self.faker.date_time_between(
 start_date=customer["created_at"], end_date="now"
 )

 # Number of line items (1-5, weighted toward fewer)
 num_items = np.random.choice([1, 2, 3, 4, 5], p=[0.4, 0.3, 0.15, 0.1, 0.05])

 # Select products for this order
 order_products = active_products.sample(
 min(num_items, len(active_products))
 )

 # Calculate totals
 subtotal = 0
 line_items = []
 for _, prod in order_products.iterrows:
 quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
 line_total = prod["price"] * quantity
 subtotal += line_total
 line_items.append(
 {
 "product_id": prod["product_id"],
 "quantity": quantity,
 "price": prod["price"],
 }
 )

 # Discounts
 discount_code = None
 discount_amount = 0
 if np.random.random < 0.2: # 20% have discount
 discount_code = np.random.choice(
 ["SAVE10", "WELCOME15", "VIP20", "FLASH25"]
 )
 discount_pct = int(discount_code[-2:]) / 100
 discount_amount = round(subtotal * discount_pct, 2)

 # Shipping
 shipping_price = round(np.random.choice([0, 5.99, 9.99, 14.99]), 2)
 if subtotal > 100:
 shipping_price = 0 # Free shipping over $100

 # Tax (simplified)
 tax_rate = 0.08
 tax = round((subtotal - discount_amount) * tax_rate, 2)

 total = round(subtotal - discount_amount + shipping_price + tax, 2)

 # Order status
 status = np.random.choice(
 ["fulfilled", "pending", "cancelled", "refunded"],
 p=[0.75, 0.15, 0.07, 0.03],
 )

 data.append(
 {
 "order_id": f"ord_{i + 1:07d}",
 "order_number": 1000 + i,
 "customer_id": customer["customer_id"],
 "email": customer["email"],
 "created_at": created_at,
 "updated_at": created_at + timedelta(hours=np.random.randint(1, 48)),
 "processed_at": created_at + timedelta(minutes=np.random.randint(1, 30)),
 "financial_status": "paid" if status != "cancelled" else "voided",
 "fulfillment_status": status,
 "subtotal_price": round(subtotal, 2),
 "total_discounts": discount_amount,
 "total_shipping": shipping_price,
 "total_tax": tax,
 "total_price": total,
 "discount_code": discount_code,
 "line_items_count": len(line_items),
 "line_items": line_items, # Nested for now, will flatten for warehouse
 "currency": "USD",
 "source_name": np.random.choice(
 ["web", "mobile", "pos"], p=[0.6, 0.3, 0.1]
 ),
 }
 )

 self._orders = pd.DataFrame(data)

 # Update customer aggregates
 order_stats = (
 self._orders.groupby("customer_id")
 .agg({"order_id": "count", "total_price": "sum"})
 .rename(columns={"order_id": "orders_count", "total_price": "total_spent"})
 )

 self._customers = self._customers.set_index("customer_id")
 self._customers.update(order_stats)
 self._customers = self._customers.reset_index

 return self._orders.copy

 def generate_line_items(
 self, orders: Optional[pd.DataFrame] = None
 ) -> pd.DataFrame:
 """Flatten line items from orders into separate table.

 Args:
 orders: Orders DataFrame with nested line_items

 Returns:
 pd.DataFrame: Flattened line items
 """
 orders = orders if orders is not None else self._orders

 if orders is None:
 raise ValueError("No orders available. Generate orders first.")

 logger.info("Flattening line items from orders")

 data = []
 line_item_id = 1
 for _, order in orders.iterrows:
 for item in order["line_items"]:
 data.append(
 {
 "line_item_id": f"li_{line_item_id:08d}",
 "order_id": order["order_id"],
 "product_id": item["product_id"],
 "quantity": item["quantity"],
 "price": item["price"],
 "total": round(item["quantity"] * item["price"], 2),
 }
 )
 line_item_id += 1

 return pd.DataFrame(data)

 def generate_full_dataset(
 self,
 num_customers: int = 200,
 num_products: int = 50,
 num_orders: int = 1000,
 ) -> Dict[str, pd.DataFrame]:
 """Generate complete e-commerce dataset with all tables.

 Args:
 num_customers: Number of customers
 num_products: Number of products
 num_orders: Number of orders

 Returns:
 Dict[str, pd.DataFrame]: Dictionary of all tables
 """
 logger.info(
 f"Generating full e-commerce dataset: "
 f"{num_customers} customers, {num_products} products, {num_orders} orders"
 )

 customers = self.generate_customers(num_customers)
 products = self.generate_products(num_products)
 orders = self.generate_orders(num_orders)
 line_items = self.generate_line_items

 # Remove nested line_items from orders for cleaner output
 orders_clean = orders.drop(columns=["line_items"])

 # Refresh customers with updated stats
 customers = self._customers.copy

 return {
 "customers": customers,
 "products": products,
 "orders": orders_clean,
 "line_items": line_items,
 }
