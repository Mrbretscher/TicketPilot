import pandas as pd
import pytest


@pytest.fixture()
def valid_ticket_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [
                "Cannot access account",
                pd.NA,
                "Billing invoice question",
                "VPN setup request",
            ],
            "body": [
                "My password reset link expired before I could use it.",
                "The mobile app crashes when I open the orders screen.",
                "Can you explain the tax line on invoice 1042?",
                "Bitte helfen Sie mir bei der Einrichtung des VPN-Zugangs.",
            ],
            "answer": [
                "Ask the user to request a new password reset link.",
                "Collect app version and device logs before escalating.",
                pd.NA,
                "Bitten Sie um Betriebssystemdetails und aktuelle Fehlermeldungen.",
            ],
            "type": ["Incident", "Problem", "Request", "Request"],
            "queue": [
                "Technical Support",
                "Product Support",
                "Billing and Payments",
                "IT Support",
            ],
            "priority": ["medium", "high", "low", "medium"],
            "language": ["en", "en", "en", "de"],
            "version": ["51", "51", "51", "51"],
            "tag_1": ["Login", "Crash", "Invoice", "VPN"],
            "tag_2": ["Password", "Mobile", pd.NA, "Access"],
            "tag_3": [pd.NA, "Orders", pd.NA, pd.NA],
            "tag_4": [pd.NA, pd.NA, pd.NA, pd.NA],
            "tag_5": [pd.NA, pd.NA, pd.NA, pd.NA],
            "tag_6": [pd.NA, pd.NA, pd.NA, pd.NA],
            "tag_7": [pd.NA, pd.NA, pd.NA, pd.NA],
            "tag_8": [pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
