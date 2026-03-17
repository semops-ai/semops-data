"""Tests for synthetic data generation."""

import pytest
import pandas as pd

from data_systems_toolkit.synthetic.ecommerce import EcommerceDataGenerator
from data_systems_toolkit.synthetic.analytics import AnalyticsDataGenerator


class TestEcommerceDataGenerator:
    """Tests for EcommerceDataGenerator."""

    def test_generate_customers(self):
        """Test customer generation."""
        gen = EcommerceDataGenerator(random_seed=42)
        customers = gen.generate_customers(num_customers=10)

        assert len(customers) == 10
        assert "customer_id" in customers.columns
        assert "email" in customers.columns
        assert customers["customer_id"].is_unique

    def test_generate_products(self):
        """Test product generation."""
        gen = EcommerceDataGenerator(random_seed=42)
        products = gen.generate_products(num_products=20)

        assert len(products) == 20
        assert "product_id" in products.columns
        assert "price" in products.columns
        assert (products["price"] > 0).all()

    def test_generate_orders(self):
        """Test order generation with referential integrity."""
        gen = EcommerceDataGenerator(random_seed=42)
        customers = gen.generate_customers(num_customers=10)
        products = gen.generate_products(num_products=20)
        orders = gen.generate_orders(num_orders=50)

        assert len(orders) == 50
        assert "order_id" in orders.columns
        assert "customer_id" in orders.columns

        # Check referential integrity
        assert orders["customer_id"].isin(customers["customer_id"]).all()

    def test_generate_full_dataset(self):
        """Test full dataset generation."""
        gen = EcommerceDataGenerator(random_seed=42)
        data = gen.generate_full_dataset(
            num_customers=10, num_products=20, num_orders=50
        )

        assert "customers" in data
        assert "products" in data
        assert "orders" in data
        assert "line_items" in data

        # Check line items reference orders and products
        assert data["line_items"]["order_id"].isin(data["orders"]["order_id"]).all()


class TestAnalyticsDataGenerator:
    """Tests for AnalyticsDataGenerator."""

    def test_generate_sessions(self):
        """Test session generation."""
        gen = AnalyticsDataGenerator(random_seed=42)
        sessions = gen.generate_sessions(num_sessions=100)

        assert len(sessions) == 100
        assert "session_id" in sessions.columns
        assert "client_id" in sessions.columns

    def test_generate_sessions_with_orders(self):
        """Test session generation tied to orders."""
        # First generate orders
        ecom_gen = EcommerceDataGenerator(random_seed=42)
        ecom_data = ecom_gen.generate_full_dataset(
            num_customers=10, num_products=20, num_orders=30
        )

        # Generate sessions tied to orders
        analytics_gen = AnalyticsDataGenerator(random_seed=42)
        sessions = analytics_gen.generate_sessions(
            num_sessions=100, orders=ecom_data["orders"]
        )

        # Check that converting sessions match orders
        converting_sessions = sessions[sessions["is_conversion"]]
        assert len(converting_sessions) == 30  # Same as num_orders
        assert converting_sessions["transaction_id"].notna().all()

    def test_generate_events(self):
        """Test event generation."""
        gen = AnalyticsDataGenerator(random_seed=42)
        sessions = gen.generate_sessions(num_sessions=10)
        events = gen.generate_events(sessions)

        assert len(events) > 0
        assert "event_id" in events.columns
        assert "session_id" in events.columns
        assert events["session_id"].isin(sessions["session_id"]).all()
