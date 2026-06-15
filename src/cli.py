"""CLI: python src/cli.py <url>"""

import sys

from predict import predict

if len(sys.argv) != 2:
    print("Usage: python src/cli.py <url>", file=sys.stderr)
    sys.exit(1)

result = predict(sys.argv[1])
icon   = "✗" if result["label"] == "Phishing" else "✓"
print(f"{icon} {result['label']}  (confidence: {result['confidence']:.1%})")
