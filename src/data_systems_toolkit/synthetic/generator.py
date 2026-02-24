"""
Main synthetic data generator orchestrator using SDV and other frameworks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union, Literal
from pathlib import Path
import joblib
from datetime import datetime

# SDV imports
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Additional generators
from faker import Faker

from ..core.logging import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
 """Main orchestrator for synthetic data generation using various approaches."""

 def __init__(self, random_seed: Optional[int] = None):
 """Initialize synthetic data generator.

 Args:
 random_seed: Random seed for reproducibility
 """
 self.random_seed = random_seed or 42
 np.random.seed(self.random_seed)

 self.faker = Faker
 self.faker.seed_instance(self.random_seed)

 self.models: Dict[str, Any] = {}

 logger.info(f"Initialized SyntheticDataGenerator with seed {self.random_seed}")

 def fit_and_generate(
 self,
 data: pd.DataFrame,
 model_type: Literal["ctgan", "tvae", "gaussian_copula"] = "gaussian_copula",
 num_rows: Optional[int] = None,
 metadata: Optional[Dict[str, Any]] = None,
 ) -> pd.DataFrame:
 """Fit a model on real data and generate synthetic data.

 Args:
 data: Real data to learn from
 model_type: Type of SDV model to use
 num_rows: Number of synthetic rows to generate (default: same as input)
 metadata: Optional metadata for better modeling

 Returns:
 pd.DataFrame: Generated synthetic data
 """
 num_rows = num_rows or len(data)

 logger.info(
 f"Fitting {model_type} model on {len(data)} rows, {len(data.columns)} columns"
 )

 # Create metadata
 if metadata is None:
 metadata_obj = SingleTableMetadata
 metadata_obj.detect_from_dataframe(data)
 else:
 metadata_obj = SingleTableMetadata.load_from_dict(metadata)

 # Initialize model
 model_classes = {
 "ctgan": CTGANSynthesizer,
 "tvae": TVAESynthesizer,
 "gaussian_copula": GaussianCopulaSynthesizer,
 }

 model = model_classes[model_type](
 metadata_obj, enforce_min_max_values=True, enforce_rounding=True
 )

 # Fit model
 model.fit(data)

 # Store model
 model_key = f"{model_type}_{datetime.now.strftime('%Y%m%d_%H%M%S')}"
 self.models[model_key] = model

 # Generate synthetic data
 logger.info(f"Generating {num_rows} synthetic rows")
 synthetic_data = model.sample(num_rows=num_rows)

 return synthetic_data

 def save_model(self, model_key: str, file_path: Union[str, Path]) -> None:
 """Save a trained model to disk.

 Args:
 model_key: Key of the model to save
 file_path: Path to save the model
 """
 if model_key not in self.models:
 raise ValueError(f"Model {model_key} not found")

 model_path = Path(file_path)
 model_path.parent.mkdir(parents=True, exist_ok=True)

 joblib.dump(self.models[model_key], model_path)
 logger.info(f"Saved model {model_key} to {model_path}")

 def load_model(
 self, file_path: Union[str, Path], model_key: Optional[str] = None
 ) -> str:
 """Load a trained model from disk.

 Args:
 file_path: Path to the saved model
 model_key: Optional key to store the model under

 Returns:
 str: Key of the loaded model
 """
 model_path = Path(file_path)
 if not model_path.exists:
 raise FileNotFoundError(f"Model file not found: {model_path}")

 model = joblib.load(model_path)

 if model_key is None:
 model_key = f"loaded_{model_path.stem}_{datetime.now.strftime('%H%M%S')}"

 self.models[model_key] = model
 logger.info(f"Loaded model as {model_key}")

 return model_key

 def generate_from_model(self, model_key: str, num_rows: int) -> pd.DataFrame:
 """Generate synthetic data from a saved model.

 Args:
 model_key: Key of the model to use
 num_rows: Number of rows to generate

 Returns:
 pd.DataFrame: Generated synthetic data
 """
 if model_key not in self.models:
 raise ValueError(f"Model {model_key} not found")

 model = self.models[model_key]
 return model.sample(num_rows=num_rows)

 def list_models(self) -> list:
 """List all available models.

 Returns:
 list: List of model keys
 """
 return list(self.models.keys)
