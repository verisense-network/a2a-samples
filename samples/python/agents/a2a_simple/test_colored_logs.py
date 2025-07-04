#!/usr/bin/env python
"""Test colored logging output"""

from app.logging_config import setup_colored_logging, get_colored_logger

# Setup colored logging
setup_colored_logging(level="DEBUG")

# Get a logger
logger = get_colored_logger(__name__)

# Test all log levels
logger.debug("This is a DEBUG message - should be in cyan")
logger.info("This is an INFO message - should be in green")
logger.warning("This is a WARNING message - should be in yellow")
logger.error("This is an ERROR message - should be in red")
logger.critical("This is a CRITICAL message - should be in magenta")

# Test with some butler-specific messages
logger.info("🔍 Discovering available agents...")
logger.info("✅ Found 26 specialized agents")
logger.warning("⚠️ Agent timeout detected")
logger.error("❌ Failed to connect to agent")

# Test multiline
logger.info("Execution plan:\n- Step 1: Analyze\n- Step 2: Implement\n- Step 3: Test")