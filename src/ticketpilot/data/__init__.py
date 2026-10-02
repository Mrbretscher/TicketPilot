"""Dataset acquisition and loading helpers."""

from ticketpilot.data.customer_support import (
    TicketDatasetAcquisitionResult,
    acquire_customer_support_tickets,
    load_ticket_csv,
)

__all__ = [
    "TicketDatasetAcquisitionResult",
    "acquire_customer_support_tickets",
    "load_ticket_csv",
]
