import json
import random
from datetime import datetime, timedelta

def generate_test_events():
    """Generate sample CloudTrail events for testing."""

    events = []

    # Suspicious deletion event
    events.append({
        "eventName": "DeleteBucket",
        "userIdentity": {"userName": "compromised-user"},
        "sourceIPAddress": "192.0.2.100",
        "eventTime": datetime.now().isoformat(),
        "errorCode": None,
        "responseElements": {"bucketName": "critical-data-bucket"}
    })

    # Failed login attempts
    for i in range(5):
        events.append({
            "eventName": "ConsoleLogin",
            "userIdentity": {"userName": f"user-{i}"},
            "sourceIPAddress": f"203.0.113.{random.randint(1,255)}",
            "eventTime": (datetime.now() - timedelta(hours=i)).isoformat(),
            "errorCode": "Failed authentication",
            "responseElements": {"ConsoleLogin": "Failure"}
        })

    # Unauthorized API calls
    events.append({
        "eventName": "CreateAccessKey",
        "userIdentity": {"userName": "junior-dev"},
        "sourceIPAddress": "198.51.100.1",
        "eventTime": datetime.now().isoformat(),
        "errorCode": "UnauthorizedAccess",
        "responseElements": None
    })

    # Root account usage
    events.append({
        "eventName": "ConsoleLogin",
        "userIdentity": {"type": "Root", "userName": "root"},
        "sourceIPAddress": "172.16.0.1",
        "eventTime": datetime.now().isoformat(),
        "errorCode": None,
        "responseElements": {"ConsoleLogin": "Success"}
    })

    # Suspicious IAM changes
    events.append({
        "eventName": "AttachUserPolicy",
        "userIdentity": {"userName": "rogue-admin"},
        "sourceIPAddress": "198.51.100.50",
        "eventTime": datetime.now().isoformat(),
        "requestParameters": {
            "userName": "backdoor-user",
            "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess"
        },
        "errorCode": None
    })

    # Security group modification
    events.append({
        "eventName": "AuthorizeSecurityGroupIngress",
        "userIdentity": {"userName": "dev-user"},
        "sourceIPAddress": "203.0.113.100",
        "eventTime": datetime.now().isoformat(),
        "requestParameters": {
            "groupId": "sg-123456",
            "ipPermissions": [{
                "ipProtocol": "-1",
                "fromPort": 0,
                "toPort": 65535,
                "ipRanges": [{"cidrIp": "0.0.0.0/0"}]
            }]
        },
        "errorCode": None
    })

    # KMS key deletion attempt
    events.append({
        "eventName": "ScheduleKeyDeletion",
        "userIdentity": {"userName": "disgruntled-employee"},
        "sourceIPAddress": "192.0.2.50",
        "eventTime": datetime.now().isoformat(),
        "requestParameters": {
            "keyId": "arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012",
            "pendingWindowInDays": 7
        },
        "errorCode": None
    })

    # Unusual data exfiltration pattern
    for i in range(10):
        events.append({
            "eventName": "GetObject",
            "userIdentity": {"userName": "data-thief"},
            "sourceIPAddress": "45.76.32.100",
            "eventTime": (datetime.now() - timedelta(minutes=i*5)).isoformat(),
            "requestParameters": {
                "bucketName": "sensitive-data",
                "key": f"customer-data/file-{i}.csv"
            },
            "errorCode": None
        })

    return events

def generate_cloudtrail_lookup_response(user_name: str, hours_back: int = 24):
    """Generate a mock CloudTrail lookup response for testing."""

    events = []
    base_time = datetime.now()

    # Generate various events for the user
    event_types = [
        ("ListBuckets", None, True),
        ("GetObject", None, True),
        ("PutObject", None, False),
        ("DeleteObject", None, False),
        ("CreateAccessKey", None, False),
        ("ConsoleLogin", "Failed authentication", True),
        ("AssumeRole", None, True),
        ("DescribeInstances", None, True),
    ]

    for i in range(min(hours_back * 2, 50)):  # Generate up to 50 events
        event_type = random.choice(event_types)
        event_time = base_time - timedelta(hours=random.uniform(0, hours_back))

        events.append({
            'EventName': event_type[0],
            'EventTime': event_time,
            'EventSource': f"{'s3' if 'Object' in event_type[0] or 'Bucket' in event_type[0] else 'iam'}.amazonaws.com",
            'Username': user_name,
            'ErrorCode': event_type[1],
            'ReadOnly': event_type[2],
            'SourceIPAddress': f"192.168.{random.randint(1,255)}.{random.randint(1,255)}"
        })

    # Sort by time
    events.sort(key=lambda x: x['EventTime'], reverse=True)

    return {'Events': events}

def create_test_iam_user():
    """Generate mock IAM user data for testing."""

    users = [
        {
            "UserName": "suspicious-user",
            "CreateDate": (datetime.now() - timedelta(days=30)).isoformat(),
            "PasswordLastUsed": (datetime.now() - timedelta(hours=2)).isoformat(),
            "Policies": ["PowerUserAccess"],
            "Groups": ["Developers"],
            "HasMFA": False
        },
        {
            "UserName": "compromised-user",
            "CreateDate": (datetime.now() - timedelta(days=365)).isoformat(),
            "PasswordLastUsed": (datetime.now() - timedelta(minutes=30)).isoformat(),
            "Policies": ["AdministratorAccess"],
            "Groups": ["Admins"],
            "HasMFA": True
        },
        {
            "UserName": "junior-dev",
            "CreateDate": (datetime.now() - timedelta(days=7)).isoformat(),
            "PasswordLastUsed": (datetime.now() - timedelta(days=1)).isoformat(),
            "Policies": ["ReadOnlyAccess"],
            "Groups": ["JuniorDevelopers"],
            "HasMFA": False
        }
    ]

    return users

if __name__ == "__main__":
    # Generate test events
    test_events = generate_test_events()

    # Save events to file
    with open("test_events.json", "w") as f:
        json.dump(test_events, f, indent=2)

    print(f"Generated {len(test_events)} test events")
    print("\nEvent types generated:")
    event_types = {}
    for event in test_events:
        event_name = event.get('eventName', 'Unknown')
        event_types[event_name] = event_types.get(event_name, 0) + 1

    for event_type, count in sorted(event_types.items()):
        print(f"  - {event_type}: {count}")

    # Generate mock IAM users
    iam_users = create_test_iam_user()
    with open("test_iam_users.json", "w") as f:
        json.dump(iam_users, f, indent=2)

    print(f"\nGenerated {len(iam_users)} test IAM users")

    # Generate mock CloudTrail lookup
    lookup_response = generate_cloudtrail_lookup_response("suspicious-user", 24)
    with open("test_cloudtrail_lookup.json", "w") as f:
        json.dump(lookup_response, f, indent=2, default=str)

    print(f"Generated {len(lookup_response['Events'])} CloudTrail lookup events")

    print("\n✅ Test data generation complete!")
    print("Files created:")
    print("  - test_events.json")
    print("  - test_iam_users.json")
    print("  - test_cloudtrail_lookup.json")
