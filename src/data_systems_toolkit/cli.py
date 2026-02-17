"""
Command-line interface for the Data Systems Toolkit.
"""

import click
from pathlib import Path

from .core.config import load_config
from .core.logging import setup_logging, get_logger


@click.group()
@click.option("--debug/--no-debug", default=False, help="Enable debug logging")
@click.pass_context
def main(ctx, debug):
    """Data Systems Toolkit - Simulate and understand enterprise data architectures."""
    ctx.ensure_object(dict)

    config = load_config()
    if debug:
        config.debug = True
        config.log_level = "DEBUG"

    setup_logging(level=config.log_level)
    ctx.obj["config"] = config


@main.command()
@click.option("--customers", default=200, help="Number of customers to generate")
@click.option("--products", default=50, help="Number of products to generate")
@click.option("--orders", default=1000, help="Number of orders to generate")
@click.option("--sessions", default=5000, help="Number of analytics sessions")
@click.option("--output-dir", default="./samples/raw", help="Output directory")
@click.pass_context
def generate(ctx, customers, products, orders, sessions, output_dir):
    """Generate synthetic e-commerce and analytics data."""
    from .synthetic.ecommerce import EcommerceDataGenerator
    from .synthetic.analytics import AnalyticsDataGenerator

    logger = get_logger(__name__)
    config = ctx.obj["config"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate e-commerce data
    logger.info("Generating e-commerce data...")
    ecom_gen = EcommerceDataGenerator(random_seed=config.synthetic_seed)
    ecom_data = ecom_gen.generate_full_dataset(
        num_customers=customers,
        num_products=products,
        num_orders=orders,
    )

    # Save e-commerce data
    for table_name, df in ecom_data.items():
        file_path = output_path / f"shopify_{table_name}.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {table_name}: {len(df)} rows -> {file_path}")

    # Generate analytics data tied to orders
    logger.info("Generating analytics data...")
    analytics_gen = AnalyticsDataGenerator(random_seed=config.synthetic_seed)
    analytics_data = analytics_gen.generate_full_dataset(
        num_sessions=sessions,
        orders=ecom_data["orders"],
        include_events=True,
    )

    # Save analytics data
    for table_name, df in analytics_data.items():
        file_path = output_path / f"ga4_{table_name}.csv"
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {table_name}: {len(df)} rows -> {file_path}")

    click.echo(f"Generated data saved to {output_path}")


@main.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output path for HTML report")
@click.option("--minimal/--full", default=False, help="Use minimal profiling mode")
@click.pass_context
def profile(ctx, file_path, output, minimal):
    """Profile a data file and generate a report."""
    from .profiling.profiler import DataProfiler

    logger = get_logger(__name__)

    profiler = DataProfiler()
    report = profiler.profile_file(file_path, minimal=minimal)

    if output is None:
        output = Path(file_path).stem + "_profile.html"

    report.to_file(output)
    logger.info(f"Profile report saved to {output}")
    click.echo(f"Profile report saved to {output}")


@main.command()
@click.pass_context
def info(ctx):
    """Show toolkit information and configuration."""
    from . import __version__

    config = ctx.obj["config"]

    click.echo(f"Data Systems Toolkit v{__version__}")
    click.echo()
    click.echo("Configuration:")
    click.echo(f"  Environment: {config.environment}")
    click.echo(f"  Debug: {config.debug}")
    click.echo(f"  DuckDB Path: {config.duckdb_path}")
    click.echo(f"  Synthetic Seed: {config.synthetic_seed}")
    click.echo(f"  Default Rows: {config.synthetic_default_rows}")


if __name__ == "__main__":
    main()
