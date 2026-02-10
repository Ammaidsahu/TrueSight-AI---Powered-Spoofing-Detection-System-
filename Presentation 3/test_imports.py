"""
Test script to verify basic application structure
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing imports...")
    
    # Test shared modules
    from src.shared.config import settings
    print("✓ Config module imported successfully")
    
    from src.shared.database import Base, User
    print("✓ Database module imported successfully")
    
    from src.shared.security import verify_password
    print("✓ Security module imported successfully")
    
    from src.shared.monitoring import setup_monitoring
    print("✓ Monitoring module imported successfully")
    
    # Test API modules
    from src.api.routes.health import router as health_router
    print("✓ Health routes imported successfully")
    
    from src.api.routes.detection import router as detection_router
    print("✓ Detection routes imported successfully")
    
    print("\n✓ All modules imported successfully!")
    print(f"Application name: {settings.APP_NAME}")
    print(f"Application version: {settings.APP_VERSION}")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()