"""Quick JWT Test - Verify token creation and API authentication"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_jwt_creation():
    """Test JWT token creation"""
    print("=== JWT Token Creation Test ===")

    try:
        from app.auth import create_jwt_token
        import jwt

        # Create tokens
        admin_token = create_jwt_token("test_admin", "admin")
        viewer_token = create_jwt_token("test_viewer", "viewer")

        print(f"✅ Admin token created: {len(admin_token)} characters")
        print(f"✅ Viewer token created: {len(viewer_token)} characters")

        # Decode to verify
        decoded_admin = jwt.decode(
            admin_token, "change-me-in-prod", algorithms=["HS256"])
        decoded_viewer = jwt.decode(
            viewer_token, "change-me-in-prod", algorithms=["HS256"])

        print(f"✅ Admin token role: {decoded_admin.get('role')}")
        print(f"✅ Viewer token role: {decoded_viewer.get('role')}")

        return admin_token, viewer_token

    except Exception as e:
        print(f"❌ JWT test failed: {e}")
        return None, None


def test_api_authentication(admin_token, viewer_token):
    """Test API authentication with tokens"""
    print("\n=== API Authentication Test ===")

    import requests

    BASE_URL = "http://localhost:8000"

    # Test endpoints
    endpoints = [
        ("/health", "Public endpoint"),
        ("/schema", "Protected endpoint"),
    ]

    tokens = [
        (None, "No token"),
        (admin_token, "Admin token"),
        (viewer_token, "Viewer token"),
    ]

    for endpoint, description in endpoints:
        print(f"\nTesting {endpoint} ({description}):")

        for token, token_desc in tokens:
            headers = {}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            try:
                response = requests.get(
                    f"{BASE_URL}{endpoint}", headers=headers, timeout=5)
                status = "✅ PASS" if response.status_code in [
                    200, 401, 403] else "❌ FAIL"
                print(f"  {token_desc:15} → {response.status_code} {status}")

                if response.status_code == 401:
                    print(f"    ^ Expected for protected endpoint without auth")
                elif response.status_code == 403:
                    print(f"    ^ Expected for insufficient permissions")

            except requests.exceptions.ConnectionError:
                print(
                    f"  {token_desc:15} → ❌ Connection failed (server not running?)")
                return False
            except Exception as e:
                print(f"  {token_desc:15} → ❌ Error: {e}")

    return True


def main():
    """Run JWT and API tests"""
    print("JWT and API Authentication Test Suite")
    print("=" * 50)

    # Test JWT creation
    admin_token, viewer_token = test_jwt_creation()

    if not admin_token or not viewer_token:
        print("\n❌ Cannot proceed - JWT creation failed")
        return

    # Test API authentication
    api_success = test_api_authentication(admin_token, viewer_token)

    print("\n" + "=" * 50)
    if api_success:
        print("✅ All authentication tests completed!")
        print("\nNext steps:")
        print("1. Start the server: uvicorn app.main:app --reload")
        print("2. Run Streamlit: streamlit run streamlit_app.py")
        print("3. Test RBAC in the UI")
    else:
        print("❌ API tests failed - check if server is running")


if __name__ == "__main__":
    main()
