"""
Web analytics synthetic data generator - GA4-style session and event data.
Simplified to focus on purchase-related sessions for e-commerce integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from faker import Faker
import hashlib

from ..core.logging import get_logger

logger = get_logger(__name__)


class AnalyticsDataGenerator:
    """Generate GA4-style web analytics data tied to e-commerce purchases."""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize analytics data generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed or 42
        np.random.seed(self.random_seed)

        self.faker = Faker()
        self.faker.seed_instance(self.random_seed)

        # Traffic source configurations
        self.traffic_sources = {
            "organic": {"medium": "organic", "source": "google", "weight": 0.35},
            "paid_search": {"medium": "cpc", "source": "google", "weight": 0.20},
            "social": {"medium": "social", "source": "facebook", "weight": 0.15},
            "email": {"medium": "email", "source": "newsletter", "weight": 0.10},
            "direct": {"medium": "(none)", "source": "(direct)", "weight": 0.15},
            "referral": {"medium": "referral", "source": "partner.com", "weight": 0.05},
        }

        # Device categories
        self.devices = {
            "desktop": 0.45,
            "mobile": 0.45,
            "tablet": 0.10,
        }

        logger.info(
            f"Initialized AnalyticsDataGenerator with seed {self.random_seed}"
        )

    def _generate_client_id(self) -> str:
        """Generate a GA4-style client ID."""
        random_part = np.random.randint(1000000000, 9999999999)
        timestamp = int(datetime.now().timestamp())
        return f"{random_part}.{timestamp}"

    def _generate_session_id(self, client_id: str, session_start: datetime) -> str:
        """Generate a session ID based on client and timestamp."""
        combined = f"{client_id}_{session_start.timestamp()}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    def generate_sessions(
        self,
        num_sessions: int = 5000,
        orders: Optional[pd.DataFrame] = None,
        conversion_rate: float = 0.03,
    ) -> pd.DataFrame:
        """Generate session data, optionally tied to orders.

        Args:
            num_sessions: Total number of sessions to generate
            orders: Optional orders DataFrame to tie conversions to
            conversion_rate: If no orders provided, what % of sessions convert

        Returns:
            pd.DataFrame: Session data
        """
        logger.info(f"Generating {num_sessions} sessions")

        # If orders provided, ensure we have sessions for each order
        order_sessions = []
        if orders is not None:
            for _, order in orders.iterrows():
                order_sessions.append(
                    {
                        "order_id": order["order_id"],
                        "customer_id": order["customer_id"],
                        "order_created_at": order["created_at"],
                        "order_total": order["total_price"],
                    }
                )

        # Calculate how many additional non-converting sessions we need
        num_converting = len(order_sessions)
        num_non_converting = num_sessions - num_converting
        if num_non_converting < 0:
            num_non_converting = 0
            logger.warning(
                f"More orders than sessions requested. Generating {num_converting} sessions."
            )

        data = []
        session_num = 0

        # Generate converting sessions (tied to orders)
        for order_info in order_sessions:
            session_num += 1
            session_data = self._create_session(
                session_num,
                is_conversion=True,
                order_info=order_info,
            )
            data.append(session_data)

        # Generate non-converting sessions
        for _ in range(num_non_converting):
            session_num += 1
            session_data = self._create_session(
                session_num,
                is_conversion=False,
            )
            data.append(session_data)

        df = pd.DataFrame(data)

        # Shuffle to mix converting and non-converting
        df = df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        return df

    def _create_session(
        self,
        session_num: int,
        is_conversion: bool,
        order_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a single session record.

        Args:
            session_num: Session sequence number
            is_conversion: Whether this session resulted in a purchase
            order_info: Order details if this is a converting session

        Returns:
            Dict: Session data
        """
        # Generate client ID
        client_id = self._generate_client_id()

        # Session timing
        if order_info:
            # Session started before order was placed
            session_start = order_info["order_created_at"] - timedelta(
                minutes=np.random.randint(5, 60)
            )
        else:
            session_start = self.faker.date_time_between(
                start_date="-90d", end_date="now"
            )

        # Session duration (converting sessions tend to be longer)
        if is_conversion:
            duration_seconds = np.random.randint(180, 1800)  # 3-30 minutes
        else:
            # Many short sessions (bounces), some longer
            if np.random.random() < 0.4:  # 40% bounce
                duration_seconds = np.random.randint(5, 30)
            else:
                duration_seconds = np.random.randint(30, 600)

        # Traffic source
        source_weights = [s["weight"] for s in self.traffic_sources.values()]
        source_names = list(self.traffic_sources.keys())
        source_key = np.random.choice(source_names, p=source_weights)
        source_config = self.traffic_sources[source_key]

        # Device
        device_weights = list(self.devices.values())
        device_names = list(self.devices.keys())
        device = np.random.choice(device_names, p=device_weights)

        # Page views (more for converting sessions)
        if is_conversion:
            pageviews = np.random.randint(4, 15)
        else:
            pageviews = np.random.randint(1, 8)

        # Events (simplified)
        events_count = pageviews + np.random.randint(0, pageviews * 2)

        session_id = self._generate_session_id(client_id, session_start)

        session = {
            "session_id": session_id,
            "client_id": client_id,
            "session_start": session_start,
            "session_end": session_start + timedelta(seconds=duration_seconds),
            "session_duration_seconds": duration_seconds,
            "session_engaged": duration_seconds > 10,
            "pageviews": pageviews,
            "events_count": events_count,
            "traffic_source": source_key,
            "medium": source_config["medium"],
            "source": source_config["source"],
            "campaign": self._get_campaign(source_key),
            "device_category": device,
            "browser": np.random.choice(
                ["Chrome", "Safari", "Firefox", "Edge"], p=[0.6, 0.2, 0.1, 0.1]
            ),
            "operating_system": self._get_os(device),
            "country": "US",
            "region": self.faker.state_abbr(),
            "city": self.faker.city(),
            "is_new_user": np.random.choice([True, False], p=[0.4, 0.6]),
            "landing_page": self._get_landing_page(source_key, is_conversion),
            "exit_page": self._get_exit_page(is_conversion),
            # Conversion fields
            "is_conversion": is_conversion,
            "transaction_id": order_info["order_id"] if order_info else None,
            "transaction_revenue": order_info["order_total"] if order_info else None,
            "customer_id": order_info["customer_id"] if order_info else None,
        }

        return session

    def _get_campaign(self, source_key: str) -> Optional[str]:
        """Get campaign name based on traffic source."""
        if source_key == "paid_search":
            return np.random.choice(
                ["brand_search", "product_search", "competitor_search"]
            )
        elif source_key == "social":
            return np.random.choice(["summer_sale", "new_arrivals", "retargeting"])
        elif source_key == "email":
            return np.random.choice(
                ["weekly_newsletter", "abandoned_cart", "welcome_series"]
            )
        return None

    def _get_os(self, device: str) -> str:
        """Get operating system based on device type."""
        if device == "mobile":
            return np.random.choice(["iOS", "Android"], p=[0.55, 0.45])
        elif device == "tablet":
            return np.random.choice(["iOS", "Android"], p=[0.7, 0.3])
        else:
            return np.random.choice(["Windows", "macOS", "Linux"], p=[0.7, 0.25, 0.05])

    def _get_landing_page(self, source_key: str, is_conversion: bool) -> str:
        """Get landing page based on traffic source and conversion status."""
        pages = {
            "organic": ["/", "/products", "/category/electronics", "/blog"],
            "paid_search": ["/products", "/sale", "/category/clothing"],
            "social": ["/sale", "/new-arrivals", "/products"],
            "email": ["/", "/sale", "/account"],
            "direct": ["/", "/products", "/cart"],
            "referral": ["/", "/products"],
        }
        return np.random.choice(pages.get(source_key, ["/"]))

    def _get_exit_page(self, is_conversion: bool) -> str:
        """Get exit page based on conversion status."""
        if is_conversion:
            return np.random.choice(
                ["/checkout/thank-you", "/order-confirmation"],
                p=[0.7, 0.3],
            )
        else:
            return np.random.choice(
                ["/", "/products", "/cart", "/checkout", "/category/electronics"],
                p=[0.3, 0.3, 0.2, 0.1, 0.1],
            )

    def generate_events(
        self,
        sessions: pd.DataFrame,
        events_per_session: int = 10,
    ) -> pd.DataFrame:
        """Generate event-level data for sessions.

        Args:
            sessions: Sessions DataFrame
            events_per_session: Average events per session

        Returns:
            pd.DataFrame: Event data
        """
        logger.info(f"Generating events for {len(sessions)} sessions")

        # Event types with weights
        event_types = {
            "page_view": 0.40,
            "scroll": 0.20,
            "click": 0.15,
            "view_item": 0.10,
            "add_to_cart": 0.05,
            "begin_checkout": 0.03,
            "purchase": 0.02,
            "search": 0.03,
            "sign_up": 0.02,
        }

        data = []
        event_id = 1

        for _, session in sessions.iterrows():
            # Number of events based on session's events_count
            num_events = session["events_count"]

            # Generate events for this session
            current_time = session["session_start"]
            time_increment = session["session_duration_seconds"] / max(num_events, 1)

            for i in range(num_events):
                # Select event type (more purchase events for converting sessions)
                if session["is_conversion"] and i == num_events - 1:
                    event_type = "purchase"
                else:
                    event_type = np.random.choice(
                        list(event_types.keys()),
                        p=list(event_types.values()),
                    )

                data.append(
                    {
                        "event_id": f"evt_{event_id:010d}",
                        "session_id": session["session_id"],
                        "client_id": session["client_id"],
                        "event_name": event_type,
                        "event_timestamp": current_time,
                        "event_params": self._get_event_params(event_type, session),
                        "page_location": self._get_page_for_event(event_type),
                    }
                )

                event_id += 1
                current_time += timedelta(seconds=time_increment)

        return pd.DataFrame(data)

    def _get_event_params(
        self, event_type: str, session: pd.Series
    ) -> Dict[str, Any]:
        """Get event parameters based on event type."""
        params = {}

        if event_type == "view_item":
            params["item_id"] = f"prod_{np.random.randint(1, 100):05d}"
            params["item_name"] = self.faker.word().title()
            params["price"] = round(np.random.uniform(10, 500), 2)

        elif event_type == "add_to_cart":
            params["item_id"] = f"prod_{np.random.randint(1, 100):05d}"
            params["quantity"] = np.random.randint(1, 3)
            params["value"] = round(np.random.uniform(10, 500), 2)

        elif event_type == "purchase":
            if session["transaction_id"]:
                params["transaction_id"] = session["transaction_id"]
                params["value"] = session["transaction_revenue"]
                params["currency"] = "USD"

        elif event_type == "search":
            params["search_term"] = self.faker.word()

        return params

    def _get_page_for_event(self, event_type: str) -> str:
        """Get page location based on event type."""
        page_map = {
            "page_view": np.random.choice(["/", "/products", "/category/electronics"]),
            "view_item": f"/products/prod_{np.random.randint(1, 100):05d}",
            "add_to_cart": f"/products/prod_{np.random.randint(1, 100):05d}",
            "begin_checkout": "/checkout",
            "purchase": "/checkout/thank-you",
            "search": "/search",
            "sign_up": "/account/register",
        }
        return page_map.get(event_type, "/")

    def generate_full_dataset(
        self,
        num_sessions: int = 5000,
        orders: Optional[pd.DataFrame] = None,
        include_events: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Generate complete analytics dataset.

        Args:
            num_sessions: Number of sessions to generate
            orders: Optional orders to tie conversions to
            include_events: Whether to generate event-level data

        Returns:
            Dict[str, pd.DataFrame]: Dictionary of analytics tables
        """
        logger.info(f"Generating full analytics dataset with {num_sessions} sessions")

        sessions = self.generate_sessions(num_sessions, orders)

        result = {"sessions": sessions}

        if include_events:
            events = self.generate_events(sessions)
            result["events"] = events

        return result
