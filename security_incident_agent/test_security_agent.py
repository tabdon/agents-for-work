import json
import time
from datetime import datetime
from security_agent import process_security_event, run_security_agent

def test_security_events():
    """Test the security agent with various event types."""

    print("=" * 70)
    print("SECURITY INCIDENT RESPONSE AGENT - TEST SUITE")
    print("=" * 70)

    # Load test events
    try:
        with open("test_events.json", "r") as f:
            test_events = json.load(f)
    except FileNotFoundError:
        print("⚠️  test_events.json not found. Running generate_test_events.py...")
        import generate_test_events
        test_events = generate_test_events.generate_test_events()

    # Test Case 1: Critical S3 Bucket Deletion
    print("\n" + "="*50)
    print("TEST 1: S3 Bucket Deletion Event")
    print("="*50)
    deletion_event = next((e for e in test_events if e['eventName'] == 'DeleteBucket'), test_events[0])
    print(f"Event: {json.dumps(deletion_event, indent=2)}")
    print("\nAgent Response:")
    print(process_security_event(deletion_event))
    time.sleep(2)  # Small delay to avoid rate limiting

    # Test Case 2: Root Account Login
    print("\n" + "="*50)
    print("TEST 2: Root Account Login Event")
    print("="*50)
    root_event = next((e for e in test_events if e.get('userIdentity', {}).get('type') == 'Root'), None)
    if root_event:
        print(f"Event: {json.dumps(root_event, indent=2)}")
        print("\nAgent Response:")
        print(process_security_event(root_event))
        time.sleep(2)

    # Test Case 3: IAM Policy Escalation
    print("\n" + "="*50)
    print("TEST 3: IAM Policy Attachment Event")
    print("="*50)
    iam_event = next((e for e in test_events if e['eventName'] == 'AttachUserPolicy'), None)
    if iam_event:
        print(f"Event: {json.dumps(iam_event, indent=2)}")
        print("\nAgent Response:")
        print(process_security_event(iam_event))
        time.sleep(2)

    # Test Case 4: Security Group Opening
    print("\n" + "="*50)
    print("TEST 4: Security Group Modification Event")
    print("="*50)
    sg_event = next((e for e in test_events if 'SecurityGroup' in e['eventName']), None)
    if sg_event:
        print(f"Event: {json.dumps(sg_event, indent=2)}")
        print("\nAgent Response:")
        print(process_security_event(sg_event))
        time.sleep(2)

    # Test Case 5: Natural Language Queries
    print("\n" + "="*50)
    print("TEST 5: Natural Language Security Queries")
    print("="*50)

    queries = [
        "Check for any failed login attempts in the last 12 hours",
        "Investigate user 'suspicious-user' activity for the last 24 hours",
        "Look up IAM details for user 'compromised-user'",
        "Are there any critical security issues I should know about?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("Response:")
        print(run_security_agent(query))
        time.sleep(2)

    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)

def test_batch_events():
    """Test processing multiple events in sequence."""

    print("\n" + "="*70)
    print("BATCH EVENT PROCESSING TEST")
    print("="*70)

    # Create a pattern of suspicious activity
    suspicious_pattern = [
        {
            "eventName": "ConsoleLogin",
            "userIdentity": {"userName": "attacker"},
            "sourceIPAddress": "45.76.32.100",
            "eventTime": datetime.now().isoformat(),
            "errorCode": "Failed authentication"
        },
        {
            "eventName": "ConsoleLogin",
            "userIdentity": {"userName": "attacker"},
            "sourceIPAddress": "45.76.32.100",
            "eventTime": datetime.now().isoformat(),
            "errorCode": "Failed authentication"
        },
        {
            "eventName": "ConsoleLogin",
            "userIdentity": {"userName": "attacker"},
            "sourceIPAddress": "45.76.32.100",
            "eventTime": datetime.now().isoformat(),
            "errorCode": None  # Successful login after failures
        },
        {
            "eventName": "CreateAccessKey",
            "userIdentity": {"userName": "attacker"},
            "sourceIPAddress": "45.76.32.100",
            "eventTime": datetime.now().isoformat(),
            "errorCode": None
        },
        {
            "eventName": "AttachUserPolicy",
            "userIdentity": {"userName": "attacker"},
            "sourceIPAddress": "45.76.32.100",
            "eventTime": datetime.now().isoformat(),
            "requestParameters": {
                "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess"
            }
        }
    ]

    print("\nSimulating attack pattern: Multiple failed logins → Success → Privilege Escalation")
    print("-" * 70)

    for i, event in enumerate(suspicious_pattern, 1):
        print(f"\nEvent {i}: {event['eventName']} - {event.get('errorCode', 'Success')}")
        response = process_security_event(event)
        print(f"Agent Action: {response[:200]}...")  # Show first 200 chars
        time.sleep(1)

    # Final investigation
    print("\n" + "-"*70)
    print("Final Investigation Query:")
    print(run_security_agent("Summarize all security issues with user 'attacker' and recommend immediate actions"))

if __name__ == "__main__":
    # Run the test suite
    test_security_events()

    # Run batch processing test
    test_batch_events()

    print("\n✅ All tests completed!")
