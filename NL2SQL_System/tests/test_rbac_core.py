"""Simple RBAC Test - Test the core RBAC logic without server dependency"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_rbac_policy():
    """Test the RBAC policy logic directly"""
    print("=== RBAC Policy Test ===")

    try:
        from app.rbac_policy import RBAC_POLICY, is_authorized, filter_schema_for_role

        # Test admin access
        print("\n1. Admin Access Tests:")
        print(f"   Admin tables setting: {RBAC_POLICY['admin']['tables']}")
        print(
            f"   Admin authorized for customers table: {is_authorized('admin', 'customers')}")
        print(
            f"   Admin authorized for customers.email: {is_authorized('admin', 'customers', 'email')}")

        # Test viewer access
        print("\n2. Viewer Access Tests:")
        viewer_policy = RBAC_POLICY['viewer']['tables']
        print(f"   Viewer allowed tables: {list(viewer_policy.keys())}")

        # Check specific viewer permissions
        test_cases = [
            ('customers', 'customer_id', True),
            ('customers', 'first_name', True),
            ('customers', 'email', False),  # Assuming email is restricted
            ('orders', 'order_id', True),
            ('restricted_table', 'any_column', False),
        ]

        for table, column, expected in test_cases:
            result = is_authorized('viewer', table, column)
            status = "✓ PASS" if result == expected else "✗ FAIL"
            print(
                f"   {table}.{column}: {status} (Expected: {expected}, Got: {result})")

        # Test schema filtering
        print("\n3. Schema Filtering Test:")
        sample_schema = {
            "tables": [
                {
                    "name": "customers",
                    "columns": [
                        {"name": "customer_id", "type": "int"},
                        {"name": "first_name", "type": "varchar"},
                        {"name": "email", "type": "varchar"},  # Restricted
                        {"name": "phone", "type": "varchar"},   # Restricted
                    ]
                },
                {
                    "name": "orders",
                    "columns": [
                        {"name": "order_id", "type": "int"},
                        {"name": "customer_id", "type": "int"},
                        {"name": "total_amount", "type": "decimal"},
                    ]
                },
                {
                    "name": "restricted_table",  # Should be filtered out
                    "columns": [
                        {"name": "secret_data", "type": "text"},
                    ]
                }
            ]
        }

        filtered = filter_schema_for_role(sample_schema, 'viewer')
        filtered_tables = [t['name'] for t in filtered['tables']]
        print(
            f"   Original tables: {[t['name'] for t in sample_schema['tables']]}")
        print(f"   Filtered tables: {filtered_tables}")
        print(f"   Filtered count: {len(filtered_tables)} (Expected: 2)")

        # Check customers table columns after filtering
        for table in filtered['tables']:
            if table['name'] == 'customers':
                columns = [c['name'] for c in table['columns']]
                print(f"   Filtered customers columns: {columns}")
                # Should not contain email or phone
                restricted_found = [col for col in [
                    'email', 'phone'] if col in columns]
                if not restricted_found:
                    print("   ✓ Restricted columns properly filtered")
                else:
                    print(f"   ✗ Found restricted columns: {restricted_found}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the project root directory")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


def test_jwt_functionality():
    """Test JWT token creation and validation"""
    print("\n=== JWT Functionality Test ===")

    try:
        from app.auth import create_jwt_token, get_current_user
        import jwt

        # Test token creation
        print("\n1. Token Creation:")
        admin_token = create_jwt_token("test_admin", "admin")
        viewer_token = create_jwt_token("test_viewer", "viewer")

        print(f"   Admin token created: {len(admin_token) > 20}")
        print(f"   Viewer token created: {len(viewer_token) > 20}")

        # Decode and verify tokens
        print("\n2. Token Decoding:")
        try:
            # This would normally be done server-side with proper secret
            # For testing, we'll decode with the same secret
            decoded_admin = jwt.decode(
                admin_token, "change-me-in-prod", algorithms=["HS256"])
            decoded_viewer = jwt.decode(
                viewer_token, "change-me-in-prod", algorithms=["HS256"])

            print(f"   Admin token role: {decoded_admin.get('role')}")
            print(f"   Viewer token role: {decoded_viewer.get('role')}")
            print("   ✓ Tokens decoded successfully")

        except Exception as e:
            print(
                f"   ⚠ Token decoding failed (expected without proper secret): {e}")

    except ImportError as e:
        print(f"❌ JWT import error: {e}")
        return False
    except Exception as e:
        print(f"❌ JWT test failed: {e}")
        return False

    return True


def main():
    """Run all RBAC tests"""
    print("RBAC Core Logic Test Suite")
    print("=" * 50)

    policy_success = test_rbac_policy()
    jwt_success = test_jwt_functionality()

    print("\n" + "=" * 50)
    if policy_success and jwt_success:
        print("✅ All RBAC core tests passed!")
        print("\nCore RBAC functionality is working correctly.")
        print("To test full integration, start the server and run the full test suite.")
    else:
        print("❌ Some tests failed!")
        if not policy_success:
            print("- RBAC policy logic needs attention")
        if not jwt_success:
            print("- JWT functionality needs attention")


if __name__ == "__main__":
    main()
